"""
Production Gateway Server

Main entry point for the distributed inference system. Provides OpenAI-compatible
API and orchestrates requests through the disaggregated prefill/decode pipeline.
"""

import asyncio
import logging
import signal
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
import json
import uuid

import aiohttp
from aiohttp import web, WSMsgType
import aiohttp.web_response
from aiohttp.web_response import Response

from ..common.models import InferenceRequest, InferenceResponse
from .disaggregated_coordinator import DisaggregatedRequestCoordinator
from .worker_client import WorkerClientPool, MockWorkerClientPool
from ..workers.prefill_worker import PrefillWorker
from ..workers.decode_worker import DecodeWorker


class ProductionGateway:
    """
    Production gateway server for distributed inference.
    
    Features:
    - OpenAI-compatible API endpoints
    - Real-time request coordination
    - Health monitoring and metrics
    - Streaming and non-streaming responses
    - Graceful shutdown and error handling
    """
    
    def __init__(self,
                 host: str = "0.0.0.0",
                 port: int = 8080,
                 model_name: str = "facebook/opt-125m",
                 use_real_vllm: bool = True):
        
        self.host = host
        self.port = port
        self.model_name = model_name
        self.use_real_vllm = use_real_vllm
        
        # Core components
        self.app: Optional[web.Application] = None
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        
        # Worker pools
        self.prefill_pool: Optional[WorkerClientPool] = None
        self.decode_pool: Optional[WorkerClientPool] = None
        
        # Request coordinator
        self.coordinator: Optional[DisaggregatedRequestCoordinator] = None
        
        # Local workers (for single-node testing)
        self.local_prefill_worker: Optional[PrefillWorker] = None
        self.local_decode_worker: Optional[DecodeWorker] = None
        
        # Metrics
        self.requests_processed = 0
        self.total_processing_time = 0.0
        self.active_requests: Dict[str, Dict] = {}
        
        logging.info(f"Initialized ProductionGateway with model {model_name}")
    
    async def initialize(self, 
                        prefill_workers: List[Dict] = None,
                        decode_workers: List[Dict] = None) -> None:
        """
        Initialize the gateway and all components.
        
        Args:
            prefill_workers: [{"host": "node1", "port": 8000}, ...]
            decode_workers: [{"host": "node2", "port": 8001}, ...]
        """
        logging.info("Initializing production gateway...")
        
        # Initialize worker pools
        if self.use_real_vllm:
            self.prefill_pool = WorkerClientPool()
            self.decode_pool = WorkerClientPool()
        else:
            self.prefill_pool = MockWorkerClientPool()
            self.decode_pool = MockWorkerClientPool()
        
        await self.prefill_pool.start()
        await self.decode_pool.start()
        
        # Add workers to pools
        if prefill_workers:
            for i, worker in enumerate(prefill_workers):
                node_id = f"prefill-{i}-{worker['host']}"
                self.prefill_pool.add_worker(node_id, worker['host'], worker['port'])
        else:
            # Start local prefill worker for testing
            await self._start_local_workers()
        
        if decode_workers:
            for i, worker in enumerate(decode_workers):
                node_id = f"decode-{i}-{worker['host']}"
                self.decode_pool.add_worker(node_id, worker['host'], worker['port'])
        
        # Initialize coordinator
        self.coordinator = DisaggregatedRequestCoordinator(
            prefill_pool=self.prefill_pool,
            decode_pool=self.decode_pool,
            node_id="gateway-coordinator",
            use_infiniband=False  # Disable for local testing
        )
        
        await self.coordinator.initialize()
        
        # Setup HTTP server
        self.app = web.Application()
        self._setup_routes()
        
        logging.info("Production gateway initialized successfully")
    
    async def _start_local_workers(self) -> None:
        """Start local workers for single-node testing."""
        logging.info("Starting local workers for testing...")
        
        # Start local prefill worker
        self.local_prefill_worker = PrefillWorker(
            node_id="local-prefill-1",
            host="127.0.0.1",
            port=18000,
            model_name=self.model_name
        )
        
        self.local_decode_worker = DecodeWorker(
            node_id="local-decode-1", 
            host="127.0.0.1",
            port=18001,
            model_name=self.model_name
        )
        
        # Start workers in background
        asyncio.create_task(self.local_prefill_worker.start_server())
        asyncio.create_task(self.local_decode_worker.start_server())
        
        # Wait for workers to start
        await asyncio.sleep(2)
        
        # Add to pools
        self.prefill_pool.add_worker("local-prefill-1", "127.0.0.1", 18000)
        self.decode_pool.add_worker("local-decode-1", "127.0.0.1", 18001)
        
        logging.info("Local workers started successfully")
    
    def _setup_routes(self) -> None:
        """Setup HTTP API routes."""
        # OpenAI-compatible endpoints
        self.app.router.add_post('/v1/completions', self.handle_completion)
        self.app.router.add_post('/v1/chat/completions', self.handle_chat_completion)
        
        # Streaming endpoints
        self.app.router.add_post('/v1/completions/stream', self.handle_completion_stream)
        
        # System endpoints
        self.app.router.add_get('/health', self.handle_health)
        self.app.router.add_get('/metrics', self.handle_metrics)
        self.app.router.add_get('/status', self.handle_status)
        
        # Debug endpoints
        self.app.router.add_get('/workers', self.handle_workers_status)
    
    async def handle_completion(self, request: web.Request) -> web.Response:
        """Handle OpenAI-compatible completion request."""
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Parse request
            data = await request.json()
            
            inference_request = InferenceRequest(
                request_id=request_id,
                prompt=data.get('prompt', ''),
                max_tokens=data.get('max_tokens', 16),
                temperature=data.get('temperature', 1.0),
                top_p=data.get('top_p', 1.0),
                stop=data.get('stop'),
                stream=data.get('stream', False)
            )
            
            # Track active request
            self.active_requests[request_id] = {
                'start_time': start_time,
                'prompt': inference_request.prompt[:100],
                'status': 'processing'
            }
            
            # Process through disaggregated system
            response_text = ""
            token_count = 0
            
            async for token in self.coordinator.process_request(inference_request):
                response_text += token
                token_count += 1
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update metrics
            self.requests_processed += 1
            self.total_processing_time += processing_time
            self.active_requests[request_id]['status'] = 'completed'
            
            # Format OpenAI-compatible response
            response = {
                "id": request_id,
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{
                    "text": response_text,
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length"
                }],
                "usage": {
                    "prompt_tokens": len(inference_request.prompt.split()),
                    "completion_tokens": token_count,
                    "total_tokens": len(inference_request.prompt.split()) + token_count
                }
            }
            
            return web.json_response(response)
            
        except Exception as e:
            logging.error(f"Completion request failed: {e}")
            self.active_requests[request_id]['status'] = 'failed'
            
            return web.json_response({
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }, status=500)
    
    async def handle_completion_stream(self, request: web.Request) -> web.StreamResponse:
        """Handle streaming completion request."""
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'text/plain'
        response.headers['Cache-Control'] = 'no-cache'
        response.headers['Connection'] = 'keep-alive'
        
        await response.prepare(request)
        
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Parse request
            data = await request.json()
            
            inference_request = InferenceRequest(
                request_id=request_id,
                prompt=data.get('prompt', ''),
                max_tokens=data.get('max_tokens', 16),
                temperature=data.get('temperature', 1.0),
                stream=True
            )
            
            # Track request
            self.active_requests[request_id] = {
                'start_time': start_time,
                'prompt': inference_request.prompt[:100],
                'status': 'streaming'
            }
            
            # Stream tokens
            token_count = 0
            async for token in self.coordinator.process_request(inference_request):
                chunk = {
                    "id": request_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [{
                        "text": token,
                        "index": 0,
                        "logprobs": None,
                        "finish_reason": None
                    }]
                }
                
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
                await response.drain()
                token_count += 1
            
            # Send completion marker
            final_chunk = {
                "id": request_id,
                "object": "text_completion", 
                "created": int(time.time()),
                "model": self.model_name,
                "choices": [{
                    "text": "",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length"
                }]
            }
            
            await response.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            
            # Update metrics
            self.requests_processed += 1
            processing_time = (time.time() - start_time) * 1000
            self.total_processing_time += processing_time
            self.active_requests[request_id]['status'] = 'completed'
            
        except Exception as e:
            logging.error(f"Streaming request failed: {e}")
            self.active_requests[request_id]['status'] = 'failed'
            
            error_chunk = {
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
            await response.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
        
        return response
    
    async def handle_chat_completion(self, request: web.Request) -> web.Response:
        """Handle OpenAI-compatible chat completion."""
        try:
            data = await request.json()
            messages = data.get('messages', [])
            
            # Convert chat messages to single prompt
            prompt = ""
            for message in messages:
                role = message.get('role', 'user')
                content = message.get('content', '')
                prompt += f"{role}: {content}\n"
            
            prompt += "assistant:"
            
            # Convert to completion request
            completion_data = {
                'prompt': prompt,
                'max_tokens': data.get('max_tokens', 16),
                'temperature': data.get('temperature', 1.0),
                'stream': data.get('stream', False)
            }
            
            # Process as regular completion
            mock_request = web.Request(
                method='POST',
                url=request.url,
                headers=request.headers,
                loop=request.loop
            )
            mock_request._payload = aiohttp.payload.JsonPayload(completion_data)
            
            return await self.handle_completion(mock_request)
            
        except Exception as e:
            return web.json_response({
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }, status=500)
    
    async def handle_health(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        healthy_prefill = len(await self.prefill_pool.get_healthy_workers())
        healthy_decode = len(await self.decode_pool.get_healthy_workers())
        
        status = {
            "status": "healthy" if healthy_prefill > 0 and healthy_decode > 0 else "unhealthy",
            "prefill_workers": healthy_prefill,
            "decode_workers": healthy_decode,
            "uptime_seconds": time.time() - getattr(self, '_start_time', time.time()),
            "requests_processed": self.requests_processed
        }
        
        return web.json_response(status)
    
    async def handle_metrics(self, request: web.Request) -> web.Response:
        """Detailed metrics endpoint."""
        coordinator_metrics = await self.coordinator.get_performance_metrics()
        
        avg_processing_time = (
            self.total_processing_time / self.requests_processed
            if self.requests_processed > 0 else 0
        )
        
        metrics = {
            "gateway_metrics": {
                "requests_processed": self.requests_processed,
                "active_requests": len(self.active_requests),
                "avg_processing_time_ms": avg_processing_time,
                "requests_per_second": self._calculate_rps()
            },
            "coordinator_metrics": coordinator_metrics,
            "worker_status": {
                "prefill_workers": len(await self.prefill_pool.get_healthy_workers()),
                "decode_workers": len(await self.decode_pool.get_healthy_workers())
            }
        }
        
        return web.json_response(metrics)
    
    async def handle_status(self, request: web.Request) -> web.Response:
        """Detailed system status."""
        status = {
            "gateway": {
                "model": self.model_name,
                "use_real_vllm": self.use_real_vllm,
                "active_requests": len(self.active_requests)
            },
            "workers": {
                "prefill": await self.prefill_pool.get_healthy_workers(),
                "decode": await self.decode_pool.get_healthy_workers()
            },
            "recent_requests": [
                {
                    "request_id": req_id,
                    "status": info["status"], 
                    "prompt_preview": info["prompt"],
                    "processing_time_ms": (time.time() - info["start_time"]) * 1000
                }
                for req_id, info in list(self.active_requests.items())[-10:]
            ]
        }
        
        return web.json_response(status)
    
    async def handle_workers_status(self, request: web.Request) -> web.Response:
        """Detailed worker status for debugging."""
        prefill_workers = []
        for node_id in await self.prefill_pool.get_healthy_workers():
            info = await self.prefill_pool.get_worker_info(node_id)
            if info:
                prefill_workers.append(info.model_dump())
        
        decode_workers = []
        for node_id in await self.decode_pool.get_healthy_workers():
            info = await self.decode_pool.get_worker_info(node_id)
            if info:
                decode_workers.append(info.model_dump())
        
        return web.json_response({
            "prefill_workers": prefill_workers,
            "decode_workers": decode_workers
        })
    
    def _calculate_rps(self) -> float:
        """Calculate requests per second."""
        uptime = time.time() - getattr(self, '_start_time', time.time())
        return self.requests_processed / uptime if uptime > 0 else 0
    
    async def start(self) -> None:
        """Start the production server."""
        self._start_time = time.time()
        
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        
        logging.info(f"Production gateway started on {self.host}:{self.port}")
        logging.info(f"Health check: http://{self.host}:{self.port}/health")
        logging.info(f"API endpoint: http://{self.host}:{self.port}/v1/completions")
    
    async def stop(self) -> None:
        """Graceful shutdown."""
        logging.info("Shutting down production gateway...")
        
        # Stop HTTP server
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        
        # Stop coordinator
        if self.coordinator:
            await self.coordinator.shutdown()
        
        # Stop worker pools
        if self.prefill_pool:
            await self.prefill_pool.stop()
        if self.decode_pool:
            await self.decode_pool.stop()
        
        # Stop local workers
        if self.local_prefill_worker:
            await self.local_prefill_worker.stop_server()
        if self.local_decode_worker:
            await self.local_decode_worker.stop_server()
        
        logging.info("Production gateway stopped")


async def main():
    """Main entry point for production server."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Production Distributed Inference Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--model", default="facebook/opt-125m", help="Model name")
    parser.add_argument("--real-vllm", action="store_true", help="Use real vLLM (vs mock)")
    parser.add_argument("--prefill-workers", help="JSON list of prefill workers")
    parser.add_argument("--decode-workers", help="JSON list of decode workers")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse worker configurations
    prefill_workers = None
    decode_workers = None
    
    if args.prefill_workers:
        prefill_workers = json.loads(args.prefill_workers)
    if args.decode_workers:
        decode_workers = json.loads(args.decode_workers)
    
    # Create and start gateway
    gateway = ProductionGateway(
        host=args.host,
        port=args.port,
        model_name=args.model,
        use_real_vllm=args.real_vllm
    )
    
    try:
        await gateway.initialize(
            prefill_workers=prefill_workers,
            decode_workers=decode_workers
        )
        await gateway.start()
        
        # Setup signal handlers for graceful shutdown
        def signal_handler():
            logging.info("Received shutdown signal")
            asyncio.create_task(gateway.stop())
        
        for sig in [signal.SIGTERM, signal.SIGINT]:
            signal.signal(sig, lambda s, f: signal_handler())
        
        # Keep running
        logging.info("Production gateway running. Press Ctrl+C to stop.")
        while True:
            await asyncio.sleep(1)
            
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt")
    finally:
        await gateway.stop()


if __name__ == "__main__":
    asyncio.run(main())