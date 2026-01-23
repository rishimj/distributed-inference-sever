"""
InfiniBand-optimized KV cache transfer for PACE ICE cluster.

Leverages RDMA (Remote Direct Memory Access) over InfiniBand for
ultra-fast cache transfers between prefill and decode nodes.
"""

import asyncio
import logging
import time
import socket
import struct
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import threading
import queue

try:
    import ucx  # UCX (Unified Communication X) for RDMA
    UCX_AVAILABLE = True
except ImportError:
    UCX_AVAILABLE = False
    logging.warning("UCX not available - falling back to TCP transfer")

try:
    import pyverbs.device as device
    import pyverbs.pd as pd  
    import pyverbs.cq as cq
    import pyverbs.qp as qp
    import pyverbs.mr as mr
    PYVERBS_AVAILABLE = True
except ImportError:
    PYVERBS_AVAILABLE = False
    logging.info("PyVerbs not available - using UCX instead")


@dataclass
class TransferStats:
    """Statistics for cache transfer performance."""
    bytes_transferred: int
    transfer_time_ms: float
    bandwidth_gbps: float
    compression_ratio: float
    rdma_used: bool
    source_node: str
    target_node: str
    timestamp: float


class InfiniBandTransferManager:
    """
    High-performance KV cache transfer over InfiniBand.
    
    Features:
    - RDMA zero-copy transfers
    - Multiple transport protocols (UCX, Verbs)
    - Automatic failover to TCP
    - Performance monitoring
    - Memory registration pooling
    """
    
    def __init__(self, 
                 node_id: str,
                 rdma_port: int = 18515,
                 tcp_fallback_port: int = 18516,
                 max_message_size: int = 1024 * 1024 * 1024,  # 1GB
                 enable_compression: bool = True):
        
        self.node_id = node_id
        self.rdma_port = rdma_port
        self.tcp_fallback_port = tcp_fallback_port
        self.max_message_size = max_message_size
        self.enable_compression = enable_compression
        
        # RDMA components
        self.ucx_context: Optional[Any] = None
        self.ucx_worker: Optional[Any] = None
        self.rdma_endpoints: Dict[str, Any] = {}  # node_id -> UCX endpoint
        
        # Transfer statistics
        self.transfer_stats: List[TransferStats] = []
        self.bandwidth_history: Dict[str, List[float]] = {}  # node_pair -> bandwidth_list
        
        # Background server
        self.server_task: Optional[asyncio.Task] = None
        self.running = False
        
        logging.info(f"Initialized InfiniBandTransferManager for {node_id}")
        logging.info(f"UCX available: {UCX_AVAILABLE}, PyVerbs available: {PYVERBS_AVAILABLE}")
    
    async def initialize(self) -> None:
        """Initialize InfiniBand/RDMA components."""
        if UCX_AVAILABLE:
            await self._initialize_ucx()
        else:
            logging.warning("UCX not available - RDMA transfers disabled")
        
        # Start background server for incoming transfers
        self.running = True
        self.server_task = asyncio.create_task(self._run_transfer_server())
        
        logging.info(f"InfiniBand transfer manager initialized for {self.node_id}")
    
    async def _initialize_ucx(self) -> None:
        """Initialize UCX for RDMA communication."""
        try:
            import ucx
            
            # UCX configuration for InfiniBand
            ucx_config = {
                'TLS': 'rc_mlx5,ud_mlx5,tcp',  # InfiniBand RC, UD, TCP fallback
                'NET_DEVICES': 'mlx5_0:1',      # InfiniBand device
                'RNDV_SCHEME': 'put_zcopy',     # Zero-copy RDMA
                'RNDV_THRESH': '8192',          # Use RDMA for messages > 8KB
                'MAX_RNDV_RAILS': '1',          # Single rail for simplicity
                'MM_TLS': 'posix',              # Memory management
                'LOG_LEVEL': 'info'
            }
            
            # Set UCX environment
            for key, value in ucx_config.items():
                import os
                os.environ[f'UCX_{key}'] = value
            
            # Initialize UCX context and worker
            self.ucx_context = ucx.create_context()
            self.ucx_worker = ucx.create_worker(self.ucx_context)
            
            logging.info("UCX initialized successfully for InfiniBand")
            
        except Exception as e:
            logging.error(f"Failed to initialize UCX: {e}")
            self.ucx_context = None
            self.ucx_worker = None
    
    async def get_node_address(self, node_name: str) -> str:
        """
        Get InfiniBand address for a node.
        PACE ICE nodes typically have predictable naming and IB addresses.
        """
        try:
            # Try to resolve hostname to InfiniBand address
            # PACE ICE InfiniBand network is typically on a separate subnet
            ib_hostname = f"{node_name}-ib"  # Common convention
            
            try:
                ib_address = socket.gethostbyname(ib_hostname)
                logging.info(f"Resolved {node_name} InfiniBand address: {ib_address}")
                return ib_address
            except socket.gaierror:
                # Fallback to regular hostname
                regular_address = socket.gethostbyname(node_name)
                logging.warning(f"No InfiniBand address for {node_name}, using {regular_address}")
                return regular_address
                
        except Exception as e:
            logging.error(f"Failed to resolve address for {node_name}: {e}")
            raise
    
    async def establish_connection(self, target_node: str) -> bool:
        """
        Establish RDMA connection to target node.
        """
        if target_node in self.rdma_endpoints:
            return True  # Already connected
        
        if not self.ucx_worker:
            logging.warning(f"UCX not available - cannot establish RDMA to {target_node}")
            return False
        
        try:
            target_address = await self.get_node_address(target_node)
            
            # Create UCX endpoint
            endpoint_address = f"{target_address}:{self.rdma_port}"
            endpoint = await self.ucx_worker.create_endpoint(endpoint_address)
            
            self.rdma_endpoints[target_node] = endpoint
            logging.info(f"RDMA connection established to {target_node} at {endpoint_address}")
            
            return True
            
        except Exception as e:
            logging.error(f"Failed to establish RDMA connection to {target_node}: {e}")
            return False
    
    async def transfer_cache_rdma(self, 
                                 target_node: str, 
                                 cache_data: bytes,
                                 request_id: str) -> bool:
        """
        Transfer KV cache using RDMA over InfiniBand.
        
        This is the high-performance path for cache transfers.
        """
        if not await self.establish_connection(target_node):
            return False
        
        endpoint = self.rdma_endpoints.get(target_node)
        if not endpoint:
            return False
        
        transfer_start = time.time()
        
        try:
            # Compress cache data if enabled
            if self.enable_compression:
                import lz4.frame
                compressed_data = lz4.frame.compress(cache_data, compression_level=1)
                compression_ratio = len(cache_data) / len(compressed_data)
                transfer_data = compressed_data
            else:
                transfer_data = cache_data
                compression_ratio = 1.0
            
            # Create transfer header
            header = struct.pack('!QQQI', 
                               len(cache_data),        # Original size
                               len(transfer_data),     # Compressed size 
                               int(self.enable_compression), # Compression flag
                               len(request_id))        # Request ID length
            
            # Send header + request_id + data
            message = header + request_id.encode('utf-8') + transfer_data
            
            # RDMA send
            await endpoint.send(message)
            
            transfer_time = (time.time() - transfer_start) * 1000
            bandwidth_gbps = (len(message) * 8) / (transfer_time / 1000) / 1e9
            
            # Record statistics
            stats = TransferStats(
                bytes_transferred=len(cache_data),
                transfer_time_ms=transfer_time,
                bandwidth_gbps=bandwidth_gbps,
                compression_ratio=compression_ratio,
                rdma_used=True,
                source_node=self.node_id,
                target_node=target_node,
                timestamp=time.time()
            )
            
            self.transfer_stats.append(stats)
            self._update_bandwidth_history(target_node, bandwidth_gbps)
            
            logging.info(f"RDMA cache transfer to {target_node}: "
                        f"{len(cache_data)} bytes in {transfer_time:.1f}ms "
                        f"({bandwidth_gbps:.2f} Gbps, compression: {compression_ratio:.1f}x)")
            
            return True
            
        except Exception as e:
            logging.error(f"RDMA transfer to {target_node} failed: {e}")
            return False
    
    async def transfer_cache_tcp(self, 
                                target_node: str, 
                                cache_data: bytes,
                                request_id: str) -> bool:
        """
        Fallback TCP transfer for cache data.
        Used when RDMA is not available.
        """
        transfer_start = time.time()
        
        try:
            target_address = await self.get_node_address(target_node)
            
            # Compress if enabled
            if self.enable_compression:
                import lz4.frame
                compressed_data = lz4.frame.compress(cache_data, compression_level=1)
                compression_ratio = len(cache_data) / len(compressed_data)
                transfer_data = compressed_data
            else:
                transfer_data = cache_data
                compression_ratio = 1.0
            
            # Connect and send
            reader, writer = await asyncio.open_connection(
                target_address, self.tcp_fallback_port
            )
            
            # Send header
            header = struct.pack('!QQQI',
                               len(cache_data),
                               len(transfer_data), 
                               int(self.enable_compression),
                               len(request_id))
            
            writer.write(header)
            writer.write(request_id.encode('utf-8'))
            writer.write(transfer_data)
            await writer.drain()
            
            writer.close()
            await writer.wait_closed()
            
            transfer_time = (time.time() - transfer_start) * 1000
            bandwidth_gbps = (len(transfer_data) * 8) / (transfer_time / 1000) / 1e9
            
            # Record statistics
            stats = TransferStats(
                bytes_transferred=len(cache_data),
                transfer_time_ms=transfer_time,
                bandwidth_gbps=bandwidth_gbps,
                compression_ratio=compression_ratio,
                rdma_used=False,
                source_node=self.node_id,
                target_node=target_node,
                timestamp=time.time()
            )
            
            self.transfer_stats.append(stats)
            self._update_bandwidth_history(target_node, bandwidth_gbps)
            
            logging.info(f"TCP cache transfer to {target_node}: "
                        f"{len(cache_data)} bytes in {transfer_time:.1f}ms "
                        f"({bandwidth_gbps:.2f} Gbps)")
            
            return True
            
        except Exception as e:
            logging.error(f"TCP transfer to {target_node} failed: {e}")
            return False
    
    async def transfer_cache(self, 
                           target_node: str, 
                           cache_data: bytes,
                           request_id: str) -> bool:
        """
        Transfer KV cache with automatic RDMA/TCP selection.
        """
        # Try RDMA first if available
        if UCX_AVAILABLE and self.ucx_worker:
            success = await self.transfer_cache_rdma(target_node, cache_data, request_id)
            if success:
                return True
            
            logging.warning(f"RDMA transfer to {target_node} failed, falling back to TCP")
        
        # Fallback to TCP
        return await self.transfer_cache_tcp(target_node, cache_data, request_id)
    
    async def _run_transfer_server(self) -> None:
        """
        Background server to receive incoming cache transfers.
        """
        # Start both RDMA and TCP servers
        rdma_task = None
        tcp_task = None
        
        try:
            if UCX_AVAILABLE and self.ucx_worker:
                rdma_task = asyncio.create_task(self._run_rdma_server())
            
            tcp_task = asyncio.create_task(self._run_tcp_server())
            
            # Wait for both servers
            tasks = [t for t in [rdma_task, tcp_task] if t]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
                
        except Exception as e:
            logging.error(f"Transfer server error: {e}")
        finally:
            if rdma_task:
                rdma_task.cancel()
            if tcp_task:
                tcp_task.cancel()
    
    async def _run_rdma_server(self) -> None:
        """Run RDMA server for incoming transfers."""
        if not self.ucx_worker:
            return
        
        try:
            # Create UCX listener
            listener = await self.ucx_worker.create_listener(
                f"0.0.0.0:{self.rdma_port}",
                self._handle_rdma_connection
            )
            
            logging.info(f"RDMA server listening on port {self.rdma_port}")
            
            # Keep server running
            while self.running:
                await asyncio.sleep(1)
                
        except Exception as e:
            logging.error(f"RDMA server error: {e}")
    
    async def _run_tcp_server(self) -> None:
        """Run TCP fallback server."""
        try:
            server = await asyncio.start_server(
                self._handle_tcp_connection,
                '0.0.0.0',
                self.tcp_fallback_port
            )
            
            logging.info(f"TCP fallback server listening on port {self.tcp_fallback_port}")
            
            async with server:
                await server.serve_forever()
                
        except Exception as e:
            logging.error(f"TCP server error: {e}")
    
    async def _handle_rdma_connection(self, endpoint) -> None:
        """Handle incoming RDMA connection."""
        try:
            # Receive message
            message = await endpoint.recv()
            await self._process_received_cache(message, rdma_used=True)
            
        except Exception as e:
            logging.error(f"RDMA connection handler error: {e}")
    
    async def _handle_tcp_connection(self, reader, writer) -> None:
        """Handle incoming TCP connection."""
        try:
            # Read all data
            data = await reader.read()
            await self._process_received_cache(data, rdma_used=False)
            
            writer.close()
            await writer.wait_closed()
            
        except Exception as e:
            logging.error(f"TCP connection handler error: {e}")
    
    async def _process_received_cache(self, data: bytes, rdma_used: bool) -> None:
        """Process received cache data."""
        try:
            # Parse header
            header_size = 32  # 8 + 8 + 8 + 4 bytes
            if len(data) < header_size:
                raise ValueError("Invalid message: too short")
            
            header = data[:header_size]
            original_size, compressed_size, compression_flag, request_id_len = struct.unpack('!QQQI', header)
            
            # Extract request ID and cache data
            request_id = data[header_size:header_size + request_id_len].decode('utf-8')
            cache_data = data[header_size + request_id_len:]
            
            # Decompress if needed
            if compression_flag:
                import lz4.frame
                decompressed_data = lz4.frame.decompress(cache_data)
            else:
                decompressed_data = cache_data
            
            # Verify size
            if len(decompressed_data) != original_size:
                raise ValueError("Size mismatch after decompression")
            
            # Store received cache (in real implementation, would inject into decode worker)
            logging.info(f"Received cache for {request_id}: {original_size} bytes "
                        f"({'RDMA' if rdma_used else 'TCP'})")
            
            # TODO: Inject cache into decode worker
            # This would call something like:
            # await self.decode_worker.inject_received_cache(request_id, decompressed_data)
            
        except Exception as e:
            logging.error(f"Failed to process received cache: {e}")
    
    def _update_bandwidth_history(self, target_node: str, bandwidth_gbps: float) -> None:
        """Update bandwidth history for performance tracking."""
        if target_node not in self.bandwidth_history:
            self.bandwidth_history[target_node] = []
        
        self.bandwidth_history[target_node].append(bandwidth_gbps)
        
        # Keep only recent measurements
        if len(self.bandwidth_history[target_node]) > 100:
            self.bandwidth_history[target_node] = self.bandwidth_history[target_node][-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get transfer performance statistics."""
        if not self.transfer_stats:
            return {'status': 'no_transfers'}
        
        recent_stats = self.transfer_stats[-50:]  # Last 50 transfers
        
        total_bytes = sum(s.bytes_transferred for s in recent_stats)
        total_time = sum(s.transfer_time_ms for s in recent_stats)
        avg_bandwidth = sum(s.bandwidth_gbps for s in recent_stats) / len(recent_stats)
        avg_compression = sum(s.compression_ratio for s in recent_stats) / len(recent_stats)
        
        rdma_transfers = sum(1 for s in recent_stats if s.rdma_used)
        tcp_transfers = len(recent_stats) - rdma_transfers
        
        return {
            'total_transfers': len(self.transfer_stats),
            'recent_transfers': len(recent_stats),
            'total_bytes_transferred': total_bytes,
            'avg_bandwidth_gbps': avg_bandwidth,
            'avg_compression_ratio': avg_compression,
            'rdma_transfers': rdma_transfers,
            'tcp_transfers': tcp_transfers,
            'rdma_success_rate': rdma_transfers / len(recent_stats) if recent_stats else 0,
            'bandwidth_by_node': {
                node: sum(bandwidths[-10:]) / len(bandwidths[-10:])
                for node, bandwidths in self.bandwidth_history.items()
                if bandwidths
            }
        }
    
    async def shutdown(self) -> None:
        """Clean shutdown of transfer manager."""
        self.running = False
        
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
        
        # Close RDMA endpoints
        for endpoint in self.rdma_endpoints.values():
            try:
                await endpoint.close()
            except:
                pass
        
        self.rdma_endpoints.clear()
        
        # Cleanup UCX resources
        if self.ucx_worker:
            self.ucx_worker = None
        if self.ucx_context:
            self.ucx_context = None
        
        logging.info(f"InfiniBand transfer manager shut down for {self.node_id}")


# Mock implementation for testing without InfiniBand hardware
class MockInfiniBandTransferManager(InfiniBandTransferManager):
    """Mock implementation for testing without InfiniBand."""
    
    async def initialize(self) -> None:
        """Mock initialization."""
        self.running = True
        logging.info(f"Mock InfiniBand manager initialized for {self.node_id}")
    
    async def get_node_address(self, node_name: str) -> str:
        """Mock node address resolution."""
        return f"192.168.100.{abs(hash(node_name)) % 200 + 10}"  # Fake IP
    
    async def transfer_cache(self, target_node: str, cache_data: bytes, request_id: str) -> bool:
        """Mock cache transfer with simulated InfiniBand performance."""
        # Simulate InfiniBand transfer time
        transfer_size_mb = len(cache_data) / (1024 * 1024)
        simulated_bandwidth_gbps = 25.0  # Typical IB bandwidth
        transfer_time_ms = (transfer_size_mb * 8) / simulated_bandwidth_gbps * 1000
        
        await asyncio.sleep(transfer_time_ms / 1000)
        
        # Record mock statistics
        stats = TransferStats(
            bytes_transferred=len(cache_data),
            transfer_time_ms=transfer_time_ms,
            bandwidth_gbps=simulated_bandwidth_gbps,
            compression_ratio=10.0,  # Assume good compression
            rdma_used=True,  # Simulate RDMA
            source_node=self.node_id,
            target_node=target_node,
            timestamp=time.time()
        )
        
        self.transfer_stats.append(stats)
        self._update_bandwidth_history(target_node, simulated_bandwidth_gbps)
        
        logging.info(f"Mock InfiniBand transfer to {target_node}: "
                    f"{len(cache_data)} bytes in {transfer_time_ms:.1f}ms "
                    f"({simulated_bandwidth_gbps} Gbps simulated)")
        
        return True