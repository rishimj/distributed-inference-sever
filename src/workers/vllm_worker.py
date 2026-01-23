"""
vLLM Worker Service - HTTP wrapper around vLLM AsyncLLMEngine.

This service provides:
1. HTTP API for inference requests
2. Health monitoring and metrics
3. KV cache state tracking
4. Resource utilization reporting

Design decisions:
- HTTP/JSON API for simplicity (can upgrade to gRPC later)
- AsyncLLMEngine for best vLLM performance
- Streaming support for long generations
- Built-in metrics and monitoring
"""

import asyncio
import json
import time
from typing import AsyncGenerator, Dict, List, Optional
from uuid import uuid4

import structlog
from aiohttp import web, ClientSession
from aiohttp.web_response import Response

# vLLM imports (these would be real imports in practice)
try:
    from vllm import AsyncLLMEngine, SamplingParams
    from vllm.utils import random_uuid
    VLLM_AVAILABLE = True
except ImportError:
    # Mock for development/testing without vLLM
    VLLM_AVAILABLE = False
    class AsyncLLMEngine:
        pass
    class SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

from ..common.models import InferenceRequest, InferenceResponse


logger = structlog.get_logger()


class MockVLLMEngine:
    """Mock vLLM engine for development/testing."""
    
    def __init__(self, model: str):
        self.model = model
        self.active_requests: Dict[str, Dict] = {}
        
    async def generate(self, prompt: str, sampling_params: SamplingParams, request_id: str):
        """Mock generation that simulates vLLM behavior."""
        self.active_requests[request_id] = {
            'prompt': prompt,
            'start_time': time.time(),
            'status': 'processing'
        }
        
        # Simulate prefill + decode time
        prefill_time = len(prompt.split()) * 0.001  # 1ms per token
        await asyncio.sleep(prefill_time)
        
        # Generate mock tokens
        tokens_to_generate = min(sampling_params.max_tokens or 100, 100)
        generated_text = ""
        
        for i in range(tokens_to_generate):
            # Simulate decode time
            await asyncio.sleep(0.02)  # 20ms per token
            
            # Generate mock token
            if i == 0:
                generated_text += " Generated response token"
            else:
                generated_text += f" {i}"
        
        # Clean up
        if request_id in self.active_requests:
            del self.active_requests[request_id]
        
        # Yield final result (mock single response)
        yield {
            'request_id': request_id,
            'text': generated_text,
            'finished': True,
            'finish_reason': 'length'
        }
    
    async def get_model_config(self):
        """Mock model config."""
        return {
            'model': self.model,
            'max_model_len': 4096,
            'max_num_seqs': 256,
            'max_num_batched_tokens': 2048
        }
    
    def get_stats(self):
        """Mock engine stats."""
        return {
            'num_requests_running': len(self.active_requests),
            'num_requests_waiting': 0,
            'num_requests_swapped': 0,
            'gpu_cache_usage_sys': 0.3,
            'cpu_cache_usage_sys': 0.1
        }


class VLLMWorkerService:
    """
    HTTP service wrapping vLLM AsyncLLMEngine.
    
    Provides REST API for inference with built-in monitoring.
    """
    
    def __init__(self, 
                 model: str,
                 host: str = "0.0.0.0",
                 port: int = 8001,
                 max_num_seqs: int = 256,
                 max_model_len: int = 4096,
                 gpu_memory_utilization: float = 0.8,
                 worker_id: Optional[str] = None):
        """
        Initialize vLLM worker service.
        
        Args:
            model: Model name/path for vLLM
            host: Host to bind HTTP server
            port: Port for HTTP server
            max_num_seqs: Max concurrent sequences (vLLM config)
            max_model_len: Max sequence length (vLLM config)  
            gpu_memory_utilization: GPU memory usage fraction
            worker_id: Unique worker identifier
        """
        self.model = model
        self.host = host
        self.port = port
        self.worker_id = worker_id or f"worker-{port}"
        
        # vLLM configuration
        self.vllm_config = {
            'model': model,
            'max_num_seqs': max_num_seqs,
            'max_model_len': max_model_len,
            'gpu_memory_utilization': gpu_memory_utilization,
            'disable_log_stats': True,  # We'll handle stats ourselves
            'enable_prefix_caching': True  # Enable KV cache optimization
        }
        
        # State tracking
        self.engine: Optional[AsyncLLMEngine] = None
        self.start_time = time.time()
        self.request_count = 0
        self.total_tokens_generated = 0
        self.active_requests: Dict[str, Dict] = {}
        
        # Performance tracking
        self.latency_history = []
        self.throughput_history = []
        
        # Cache affinity tracking for routing
        self.processed_prefixes: Dict[str, float] = {}  # prefix_hash -> last_processed_time
        self.prefix_cleanup_interval = 300  # 5 minutes
        
    async def start_engine(self) -> None:
        """Initialize and start the vLLM engine."""
        logger.info("Starting vLLM engine", model=self.model, worker_id=self.worker_id)
        
        try:
            if VLLM_AVAILABLE:
                # Real vLLM engine
                from vllm.engine.arg_utils import AsyncEngineArgs
                
                engine_args = AsyncEngineArgs(
                    model=self.vllm_config['model'],
                    max_num_seqs=self.vllm_config['max_num_seqs'],
                    max_model_len=self.vllm_config['max_model_len'],
                    gpu_memory_utilization=self.vllm_config['gpu_memory_utilization'],
                    disable_log_stats=self.vllm_config['disable_log_stats'],
                    enable_prefix_caching=self.vllm_config['enable_prefix_caching']
                )
                
                self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            else:
                # Mock engine for development
                logger.warning("vLLM not available, using mock engine")
                self.engine = MockVLLMEngine(self.model)
            
            logger.info("vLLM engine started successfully", worker_id=self.worker_id)
            
        except Exception as e:
            logger.error("Failed to start vLLM engine", error=str(e), worker_id=self.worker_id)
            raise
    
    async def generate_response(self, request: InferenceRequest) -> AsyncGenerator[Dict, None]:
        """
        Generate response using vLLM engine.
        
        Args:
            request: Inference request
            
        Yields:
            Dictionary with response chunks
        """
        if not self.engine:
            raise RuntimeError("vLLM engine not initialized")
        
        request_id = request.request_id
        start_time = time.time()
        
        # Track request
        self.active_requests[request_id] = {
            'start_time': start_time,
            'prompt_length': len(request.prompt),
            'max_tokens': request.max_tokens,
            'status': 'processing'
        }
        
        try:
            # Create sampling parameters
            sampling_params = SamplingParams(
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                stream=True  # Enable streaming
            )
            
            # Generate with vLLM
            tokens_generated = 0
            generated_text = ""
            
            async for output in self.engine.generate(request.prompt, sampling_params, request_id):
                # Extract generated content
                if hasattr(output, 'outputs') and output.outputs:
                    # Real vLLM output format
                    new_text = output.outputs[0].text[len(generated_text):]
                    generated_text = output.outputs[0].text
                    tokens_generated = len(output.outputs[0].token_ids) if hasattr(output.outputs[0], 'token_ids') else len(generated_text.split())
                    finished = output.finished
                else:
                    # Mock output format
                    new_text = output.get('text', '')
                    generated_text = new_text
                    tokens_generated = len(new_text.split())
                    finished = output.get('finished', False)
                
                # Yield streaming response
                response_chunk = {
                    'request_id': request_id,
                    'text_delta': new_text,
                    'total_text': generated_text,
                    'tokens_generated': tokens_generated,
                    'finished': finished,
                    'worker_id': self.worker_id
                }
                
                yield response_chunk
                
                if finished:
                    break
            
            # Record completion metrics
            total_time = time.time() - start_time
            self.latency_history.append(total_time * 1000)  # Convert to ms
            
            if tokens_generated > 0:
                tps = tokens_generated / total_time
                self.throughput_history.append(tps)
            
            self.total_tokens_generated += tokens_generated
            self.request_count += 1
            
            # Track prefix for cache affinity
            prefix_hash = self._compute_prefix_hash(request.prompt)
            self.processed_prefixes[prefix_hash] = time.time()
            
            logger.info(
                "Request completed",
                request_id=request_id,
                worker_id=self.worker_id,
                total_time_ms=total_time * 1000,
                tokens_generated=tokens_generated,
                tps=tokens_generated / total_time if total_time > 0 else 0
            )
            
        except Exception as e:
            logger.error(
                "Generation failed",
                request_id=request_id,
                worker_id=self.worker_id,
                error=str(e)
            )
            raise
        
        finally:
            # Clean up request tracking
            if request_id in self.active_requests:
                del self.active_requests[request_id]
    
    def _compute_prefix_hash(self, prompt: str, prefix_length: int = 512) -> str:
        """Compute prefix hash for cache tracking."""
        import hashlib
        prefix = prompt[:prefix_length]
        return hashlib.sha256(prefix.encode('utf-8')).hexdigest()
    
    async def get_health_status(self) -> Dict:
        """Get worker health and status information."""
        uptime = time.time() - self.start_time
        
        # Get engine stats
        engine_stats = {}
        if self.engine and hasattr(self.engine, 'get_stats'):
            try:
                engine_stats = self.engine.get_stats()
            except Exception as e:
                logger.warning("Failed to get engine stats", error=str(e))
        
        # Calculate performance metrics
        avg_latency = sum(self.latency_history[-100:]) / len(self.latency_history[-100:]) if self.latency_history else 0
        avg_throughput = sum(self.throughput_history[-100:]) / len(self.throughput_history[-100:]) if self.throughput_history else 0
        
        return {
            'worker_id': self.worker_id,
            'status': 'healthy' if self.engine else 'initializing',
            'uptime_seconds': uptime,
            'model': self.model,
            
            # Request statistics
            'total_requests': self.request_count,
            'active_requests': len(self.active_requests),
            'total_tokens_generated': self.total_tokens_generated,
            
            # Performance metrics
            'avg_latency_ms': avg_latency,
            'avg_throughput_tps': avg_throughput,
            'current_load': min(1.0, len(self.active_requests) / max(1, self.vllm_config.get('max_num_seqs', 256))),
            
            # Engine stats
            'engine_stats': engine_stats,
            
            # Cache info for routing
            'processed_prefixes_count': len(self.processed_prefixes),
            'cache_retention_window_sec': self.prefix_cleanup_interval
        }
    
    async def get_processed_prefixes(self) -> Dict[str, float]:
        """Get recently processed prefixes for cache-aware routing."""
        # Clean up old prefixes
        current_time = time.time()
        cutoff_time = current_time - self.prefix_cleanup_interval
        
        # Remove expired prefixes
        expired_prefixes = [
            prefix_hash for prefix_hash, timestamp in self.processed_prefixes.items()
            if timestamp < cutoff_time
        ]
        
        for prefix_hash in expired_prefixes:
            del self.processed_prefixes[prefix_hash]
        
        return self.processed_prefixes.copy()
    
    async def cleanup_cache_tracking(self) -> None:
        """Background task to clean up old cache tracking data."""
        while True:
            try:
                await self.get_processed_prefixes()  # This handles cleanup
                await asyncio.sleep(60)  # Clean up every minute
            except Exception as e:
                logger.error("Cache cleanup failed", error=str(e))
                await asyncio.sleep(60)


async def create_app(worker_service: VLLMWorkerService) -> web.Application:
    """Create aiohttp application with routes."""
    
    async def health_handler(request):
        """Health check endpoint.""" 
        health_status = await worker_service.get_health_status()
        return web.json_response(health_status)
    
    async def generate_handler(request):
        """Main inference endpoint."""
        try:
            # Parse request
            request_data = await request.json()
            
            # Create inference request
            inference_request = InferenceRequest(
                prompt=request_data['prompt'],
                max_tokens=request_data.get('max_tokens', 100),
                temperature=request_data.get('temperature', 0.7),
                top_p=request_data.get('top_p', 0.9)
            )
            
            # Check if streaming requested
            stream = request_data.get('stream', False)
            
            if stream:
                # Streaming response
                response = web.StreamResponse()
                response.headers['Content-Type'] = 'text/plain; charset=utf-8'
                await response.prepare(request)
                
                async for chunk in worker_service.generate_response(inference_request):
                    chunk_json = json.dumps(chunk) + '\n'
                    await response.write(chunk_json.encode('utf-8'))
                
                await response.write_eof()
                return response
            
            else:
                # Non-streaming response - collect all chunks
                full_response = None
                async for chunk in worker_service.generate_response(inference_request):
                    full_response = chunk
                
                return web.json_response(full_response)
        
        except Exception as e:
            logger.error("Generation request failed", error=str(e))
            return web.json_response(
                {'error': str(e)}, 
                status=500
            )
    
    async def prefixes_handler(request):
        """Get processed prefixes for cache-aware routing."""
        prefixes = await worker_service.get_processed_prefixes()
        return web.json_response({
            'processed_prefixes': prefixes,
            'worker_id': worker_service.worker_id
        })
    
    # Create application
    app = web.Application()
    
    # Add routes
    app.router.add_get('/health', health_handler)
    app.router.add_post('/generate', generate_handler)
    app.router.add_get('/prefixes', prefixes_handler)
    
    return app


async def run_worker_service(model: str, 
                           host: str = "0.0.0.0", 
                           port: int = 8001,
                           **kwargs) -> None:
    """
    Run vLLM worker service.
    
    Args:
        model: vLLM model name/path
        host: Host to bind to
        port: Port to bind to
        **kwargs: Additional vLLM configuration
    """
    # Create worker service
    worker_service = VLLMWorkerService(
        model=model,
        host=host,
        port=port,
        **kwargs
    )
    
    # Start vLLM engine
    await worker_service.start_engine()
    
    # Start background cleanup task
    cleanup_task = asyncio.create_task(worker_service.cleanup_cache_tracking())
    
    # Create web application
    app = await create_app(worker_service)
    
    # Start HTTP server
    logger.info(
        "Starting vLLM worker server",
        host=host,
        port=port,
        worker_id=worker_service.worker_id,
        model=model
    )
    
    try:
        # Run the server
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, host, port)
        await site.start()
        
        logger.info(f"vLLM worker running at http://{host}:{port}")
        
        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down worker service")
        
    finally:
        cleanup_task.cancel()
        if runner:
            await runner.cleanup()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="vLLM Worker Service")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--max-num-seqs", type=int, default=256, help="Max concurrent sequences")
    parser.add_argument("--max-model-len", type=int, default=4096, help="Max model length")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8, help="GPU memory utilization")
    
    args = parser.parse_args()
    
    # Run the service
    asyncio.run(run_worker_service(
        model=args.model,
        host=args.host,
        port=args.port,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization
    ))