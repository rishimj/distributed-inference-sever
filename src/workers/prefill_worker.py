"""
Specialized vLLM worker for prefill phase processing.

Optimized for:
- High parallel computation (batch prefill processing)
- KV cache extraction and serialization
- First token generation
- Cache transfer preparation
"""

import asyncio
import logging
import time
import pickle
import lz4.frame
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import struct

import aiohttp
from aiohttp import web
import aiohttp.web_response
from aiohttp.web_response import Response

try:
    from vllm import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    logging.warning("vLLM not available, using mock implementation")

from ..common.models import InferenceRequest


@dataclass
class PrefillResult:
    """Result from prefill processing including KV cache data."""
    request_id: str
    first_token: str
    kv_cache_data: bytes  # Serialized KV cache
    prompt_tokens: int
    cache_size_bytes: int
    processing_time_ms: float
    cache_hash: str  # For integrity checking


class KVCacheSerializer:
    """
    Handles serialization and compression of KV cache data.
    Optimized for transfer between prefill and decode nodes.
    """
    
    def __init__(self, compression_level: int = 1):
        self.compression_level = compression_level
        
    def serialize_cache(self, kv_cache: Any) -> bytes:
        """
        Serialize KV cache tensors to bytes with compression.
        
        Format: [header][compressed_data]
        Header: version(4) + original_size(8) + compressed_size(8) + checksum(4)
        """
        try:
            # Serialize the cache object
            raw_data = pickle.dumps(kv_cache, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Compress using LZ4 for speed
            compressed_data = lz4.frame.compress(
                raw_data, 
                compression_level=self.compression_level
            )
            
            # Create header
            version = 1
            original_size = len(raw_data)
            compressed_size = len(compressed_data)
            checksum = hash(raw_data) & 0xFFFFFFFF  # 32-bit checksum
            
            header = struct.pack('!IQQQ', version, original_size, compressed_size, checksum & 0xFFFFFFFF)
            
            return header + compressed_data
            
        except Exception as e:
            logging.error(f"Cache serialization failed: {e}")
            raise
    
    def deserialize_cache(self, cache_bytes: bytes) -> Any:
        """
        Deserialize KV cache from bytes with decompression and validation.
        """
        try:
            # Parse header (4 + 8 + 8 + 8 = 28 bytes for IQQQ format)
            header_size = 28  # 4 + 8 + 8 + 8
            if len(cache_bytes) < header_size:
                raise ValueError("Invalid cache data: too short")
                
            header = cache_bytes[:header_size]
            compressed_data = cache_bytes[header_size:]
            
            version, original_size, compressed_size, expected_checksum = struct.unpack('!IQQQ', header)
            
            if version != 1:
                raise ValueError(f"Unsupported cache version: {version}")
            
            if len(compressed_data) != compressed_size:
                raise ValueError("Cache data size mismatch")
            
            # Decompress
            raw_data = lz4.frame.decompress(compressed_data)
            
            if len(raw_data) != original_size:
                raise ValueError("Decompressed size mismatch")
            
            # Verify checksum
            actual_checksum = hash(raw_data) & 0xFFFFFFFF
            if actual_checksum != expected_checksum:
                raise ValueError("Cache data checksum mismatch")
            
            # Deserialize
            kv_cache = pickle.loads(raw_data)
            
            return kv_cache
            
        except Exception as e:
            logging.error(f"Cache deserialization failed: {e}")
            raise


class PrefillWorker:
    """
    Specialized worker for prefill phase processing.
    
    Optimized for:
    - High parallel computation (multiple prompts in batch)
    - KV cache extraction and transfer preparation
    - First token generation
    - Resource efficiency for compute-heavy workloads
    """
    
    def __init__(self, 
                 node_id: str,
                 host: str = "0.0.0.0",
                 port: int = 8000,
                 model_name: str = "facebook/opt-125m",
                 max_num_seqs: int = 16,  # Higher batch for prefill
                 max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.9):
        
        self.node_id = node_id
        self.host = host
        self.port = port
        self.model_name = model_name
        
        # Prefill-optimized configuration
        self.engine_config = {
            'max_num_seqs': max_num_seqs,
            'max_model_len': max_model_len,
            'gpu_memory_utilization': gpu_memory_utilization,
            'max_num_batched_tokens': max_num_seqs * 512,  # Large batch tokens
            'enable_prefix_caching': True,
            'trust_remote_code': False,
        }
        
        # Components
        self.engine: Optional[AsyncLLMEngine] = None
        self.cache_serializer = KVCacheSerializer()
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        
        # Metrics
        self.requests_processed = 0
        self.total_processing_time = 0.0
        self.cache_extractions = 0
        self.cache_transfer_bytes = 0
        
        # Active requests tracking
        self.active_requests: Dict[str, Dict] = {}
        
        logging.info(f"Initialized PrefillWorker {node_id} with config: {self.engine_config}")
    
    async def initialize_engine(self) -> None:
        """Initialize the vLLM engine with prefill-optimized settings."""
        if not VLLM_AVAILABLE:
            logging.warning("vLLM not available, using mock engine")
            self.engine = MockPrefillEngine(self.model_name, **self.engine_config)
            return
            
        try:
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                **self.engine_config
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logging.info(f"Initialized vLLM engine for prefill with model {self.model_name}")
            
        except Exception as e:
            logging.error(f"Failed to initialize vLLM engine: {e}")
            logging.info("Falling back to mock engine")
            self.engine = MockPrefillEngine(self.model_name, **self.engine_config)
    
    async def process_prefill(self, request: InferenceRequest) -> PrefillResult:
        """
        Process prefill phase: prompt → KV cache + first token.
        
        This is the core prefill operation that:
        1. Processes the prompt through attention layers
        2. Extracts the resulting KV cache
        3. Generates the first token
        4. Prepares cache for transfer to decode node
        """
        start_time = time.time()
        
        try:
            # Track active request
            self.active_requests[request.request_id] = {
                'start_time': start_time,
                'prompt_length': len(request.prompt),
                'status': 'processing'
            }
            
            # Configure sampling for just first token
            if VLLM_AVAILABLE:
                from vllm.sampling_params import SamplingParams
            else:
                # Mock SamplingParams for testing
                class SamplingParams:
                    def __init__(self, **kwargs):
                        self.__dict__.update(kwargs)
                        
            sampling_params = SamplingParams(
                max_tokens=1,  # Only generate first token in prefill
                temperature=request.temperature,
                top_p=request.top_p if hasattr(request, 'top_p') else 1.0,
                use_beam_search=False,  # Greedy for prefill
                include_stop_str_in_output=False
            )
            
            # Process through vLLM prefill
            results = await self.engine.generate(
                request.prompt,
                sampling_params,
                request_id=request.request_id
            )
            
            # Extract first token
            first_token = ""
            for result in results:
                if result.outputs:
                    first_token = result.outputs[0].text
                    break
            
            # Extract KV cache from engine
            kv_cache = await self.extract_kv_cache(request.request_id)
            
            # Serialize cache for transfer
            serialized_cache = self.cache_serializer.serialize_cache(kv_cache)
            
            # Generate cache hash for integrity
            cache_hash = str(hash(serialized_cache) & 0xFFFFFFFF)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update metrics
            self.requests_processed += 1
            self.total_processing_time += processing_time
            self.cache_extractions += 1
            self.cache_transfer_bytes += len(serialized_cache)
            
            # Clean up active request
            self.active_requests[request.request_id]['status'] = 'completed'
            
            result = PrefillResult(
                request_id=request.request_id,
                first_token=first_token,
                kv_cache_data=serialized_cache,
                prompt_tokens=len(request.prompt.split()),  # Rough estimate
                cache_size_bytes=len(serialized_cache),
                processing_time_ms=processing_time,
                cache_hash=cache_hash
            )
            
            logging.info(f"Prefill completed for {request.request_id}: "
                        f"token='{first_token}', cache_size={len(serialized_cache)} bytes, "
                        f"time={processing_time:.1f}ms")
            
            return result
            
        except Exception as e:
            self.active_requests[request.request_id]['status'] = 'failed'
            logging.error(f"Prefill processing failed for {request.request_id}: {e}")
            raise
    
    async def extract_kv_cache(self, request_id: str) -> Any:
        """
        Extract KV cache from vLLM engine for the given request.
        
        This is where we interface with vLLM internals to get the
        computed key-value cache that can be transferred to decode nodes.
        """
        try:
            # Note: This is a simplified interface
            # Real implementation would need to access vLLM internals
            if hasattr(self.engine, 'get_kv_cache'):
                return await self.engine.get_kv_cache(request_id)
            else:
                # Mock implementation - in reality this would extract actual tensors
                return {
                    'request_id': request_id,
                    'cache_blocks': f"mock_cache_data_for_{request_id}",
                    'sequence_length': 100,
                    'model_name': self.model_name
                }
                
        except Exception as e:
            logging.error(f"Failed to extract KV cache for {request_id}: {e}")
            raise
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get prefill worker performance metrics."""
        avg_processing_time = (
            self.total_processing_time / self.requests_processed 
            if self.requests_processed > 0 else 0
        )
        
        avg_cache_size = (
            self.cache_transfer_bytes / self.cache_extractions
            if self.cache_extractions > 0 else 0
        )
        
        return {
            'node_id': self.node_id,
            'worker_type': 'prefill',
            'requests_processed': self.requests_processed,
            'active_requests': len(self.active_requests),
            'avg_processing_time_ms': avg_processing_time,
            'total_cache_extractions': self.cache_extractions,
            'avg_cache_size_bytes': avg_cache_size,
            'total_cache_transfer_gb': self.cache_transfer_bytes / (1024**3),
            'engine_config': self.engine_config,
            'status': 'healthy'
        }
    
    # HTTP API Implementation
    async def handle_prefill_request(self, request: web.Request) -> web.Response:
        """Handle HTTP prefill request."""
        try:
            data = await request.json()
            inference_request = InferenceRequest(**data)
            
            result = await self.process_prefill(inference_request)
            
            return web.json_response({
                'request_id': result.request_id,
                'first_token': result.first_token,
                'kv_cache_data': result.kv_cache_data.hex(),  # Hex encode for JSON
                'prompt_tokens': result.prompt_tokens,
                'cache_size_bytes': result.cache_size_bytes,
                'processing_time_ms': result.processing_time_ms,
                'cache_hash': result.cache_hash
            })
            
        except Exception as e:
            logging.error(f"Prefill request failed: {e}")
            return web.json_response(
                {'error': str(e)}, 
                status=500
            )
    
    async def handle_health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        metrics = await self.get_metrics()
        return web.json_response({
            'status': 'healthy',
            'metrics': metrics
        })
    
    async def handle_metrics(self, request: web.Request) -> web.Response:
        """Metrics endpoint."""
        metrics = await self.get_metrics()
        return web.json_response(metrics)
    
    async def start_server(self) -> None:
        """Start the HTTP server."""
        # Initialize engine first
        await self.initialize_engine()
        
        # Setup HTTP application
        self.app = web.Application()
        
        # Add routes
        self.app.router.add_post('/prefill', self.handle_prefill_request)
        self.app.router.add_get('/health', self.handle_health_check)
        self.app.router.add_get('/metrics', self.handle_metrics)
        
        # Start server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        
        logging.info(f"PrefillWorker {self.node_id} started on {self.host}:{self.port}")
    
    async def stop_server(self) -> None:
        """Stop the HTTP server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        
        logging.info(f"PrefillWorker {self.node_id} stopped")


class MockPrefillEngine:
    """Mock implementation for testing without vLLM."""
    
    def __init__(self, model_name: str, **config):
        self.model_name = model_name
        self.config = config
        
    async def generate(self, prompt: str, sampling_params, request_id: str):
        """Mock generation - returns fake first token."""
        await asyncio.sleep(0.1)  # Simulate processing time
        
        # Mock result object
        class MockResult:
            def __init__(self):
                self.outputs = [MockOutput()]
        
        class MockOutput:
            def __init__(self):
                self.text = "Hello"  # Mock first token
                
        return [MockResult()]
    
    async def get_kv_cache(self, request_id: str):
        """Mock KV cache extraction."""
        return {
            'request_id': request_id,
            'mock_cache_data': f"cache_for_{request_id}",
            'model': self.model_name,
            'size_estimate': 1024 * 1024  # 1MB mock cache
        }


async def main():
    """Main entry point for running prefill worker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Prefill Worker for Disaggregated Inference")
    parser.add_argument("--node-id", required=True, help="Unique node identifier")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--model", default="facebook/opt-125m", help="Model name")
    parser.add_argument("--max-seqs", type=int, default=16, help="Max concurrent sequences")
    parser.add_argument("--max-len", type=int, default=4096, help="Max sequence length")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start worker
    worker = PrefillWorker(
        node_id=args.node_id,
        host=args.host,
        port=args.port,
        model_name=args.model,
        max_num_seqs=args.max_seqs,
        max_model_len=args.max_len
    )
    
    try:
        await worker.start_server()
        
        # Keep running
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Shutting down prefill worker...")
    finally:
        await worker.stop_server()


if __name__ == "__main__":
    asyncio.run(main())