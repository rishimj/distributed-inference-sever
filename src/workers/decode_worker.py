"""
Specialized vLLM worker for decode phase processing.

Optimized for:
- Sequential token generation with KV cache reuse
- High memory bandwidth utilization
- Many concurrent long-running sequences
- Cache injection and continuation
"""

import asyncio
import logging
import time
import struct
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass

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
from .prefill_worker import PrefillResult, KVCacheSerializer


@dataclass
class DecodeRequest:
    """Request for decode phase processing."""
    inference_request: InferenceRequest
    prefill_result: PrefillResult


@dataclass
class DecodeResult:
    """Result from decode phase processing."""
    request_id: str
    generated_text: str
    tokens_generated: int
    total_processing_time_ms: float
    decode_time_ms: float
    cache_hit: bool
    finish_reason: str


class DecodeWorker:
    """
    Specialized worker for decode phase processing.
    
    Optimized for:
    - Sequential token generation from KV cache
    - High memory bandwidth utilization
    - Many concurrent sequences (32-64+)
    - Long-running generation tasks
    """
    
    def __init__(self, 
                 node_id: str,
                 host: str = "0.0.0.0",
                 port: int = 8001,
                 model_name: str = "facebook/opt-125m",
                 max_num_seqs: int = 64,  # Higher concurrency for decode
                 max_model_len: int = 8192,  # Support longer sequences
                 gpu_memory_utilization: float = 0.85):  # Leave room for KV cache
        
        self.node_id = node_id
        self.host = host
        self.port = port
        self.model_name = model_name
        
        # Decode-optimized configuration
        self.engine_config = {
            'max_num_seqs': max_num_seqs,
            'max_model_len': max_model_len,
            'gpu_memory_utilization': gpu_memory_utilization,
            'max_num_batched_tokens': 4096,  # Moderate batch size for memory
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
        self.total_decode_time = 0.0
        self.total_tokens_generated = 0
        self.cache_injections = 0
        self.cache_hits = 0
        
        # Active sequences tracking
        self.active_sequences: Dict[str, Dict] = {}
        self.injected_caches: Dict[str, Any] = {}  # request_id -> cache_data
        
        logging.info(f"Initialized DecodeWorker {node_id} with config: {self.engine_config}")
    
    async def initialize_engine(self) -> None:
        """Initialize the vLLM engine with decode-optimized settings."""
        if not VLLM_AVAILABLE:
            logging.warning("vLLM not available, using mock engine")
            self.engine = MockDecodeEngine(self.model_name, **self.engine_config)
            return
            
        try:
            engine_args = AsyncEngineArgs(
                model=self.model_name,
                **self.engine_config
            )
            
            self.engine = AsyncLLMEngine.from_engine_args(engine_args)
            logging.info(f"Initialized vLLM engine for decode with model {self.model_name}")
            
        except Exception as e:
            logging.error(f"Failed to initialize vLLM engine: {e}")
            logging.info("Falling back to mock engine")
            self.engine = MockDecodeEngine(self.model_name, **self.engine_config)
    
    async def inject_kv_cache(self, request_id: str, cache_data: bytes) -> bool:
        """
        Inject KV cache into the engine for continued generation.
        
        This is the critical operation that allows continuation from prefill.
        """
        try:
            # Deserialize cache
            kv_cache = self.cache_serializer.deserialize_cache(cache_data)
            
            # Inject into engine
            # Note: This is a simplified interface - real implementation
            # would need to interface with vLLM internals to inject cache
            if hasattr(self.engine, 'inject_kv_cache'):
                success = await self.engine.inject_kv_cache(request_id, kv_cache)
            else:
                # Mock injection
                self.injected_caches[request_id] = kv_cache
                success = True
            
            if success:
                self.cache_injections += 1
                logging.info(f"Successfully injected KV cache for {request_id}")
            else:
                logging.warning(f"Failed to inject KV cache for {request_id}")
            
            return success
            
        except Exception as e:
            logging.error(f"Cache injection failed for {request_id}: {e}")
            return False
    
    async def continue_generation(self, decode_request: DecodeRequest) -> AsyncGenerator[str, None]:
        """
        Continue generation from prefill result using injected KV cache.
        
        This streams tokens as they're generated, allowing real-time response.
        """
        request = decode_request.inference_request
        prefill_result = decode_request.prefill_result
        
        start_time = time.time()
        decode_start_time = time.time()
        
        try:
            # Track active sequence
            self.active_sequences[request.request_id] = {
                'start_time': start_time,
                'prefill_time_ms': prefill_result.processing_time_ms,
                'tokens_generated': 0,
                'status': 'generating'
            }
            
            # Inject KV cache first
            cache_injected = await self.inject_kv_cache(
                request.request_id, 
                prefill_result.kv_cache_data
            )
            
            if not cache_injected:
                raise Exception("Failed to inject KV cache")
            
            self.cache_hits += 1
            
            # Configure sampling for remaining tokens
            remaining_tokens = max(1, request.max_tokens - 1)  # Minus prefill token
            
            if VLLM_AVAILABLE:
                from vllm.sampling_params import SamplingParams
            else:
                # Mock SamplingParams for testing
                class SamplingParams:
                    def __init__(self, **kwargs):
                        self.__dict__.update(kwargs)
                        
            sampling_params = SamplingParams(
                max_tokens=remaining_tokens,
                temperature=request.temperature,
                top_p=request.top_p if hasattr(request, 'top_p') else 1.0,
                stream=True,  # Enable streaming
                include_stop_str_in_output=False
            )
            
            # Yield the first token from prefill
            yield prefill_result.first_token
            
            decode_start_time = time.time()
            
            # Continue generation from cache
            token_count = 0
            async for output in self.engine.generate(
                "",  # Empty prompt since we're continuing from cache
                sampling_params,
                request_id=request.request_id,
                continue_from_cache=True  # Custom parameter for cache continuation
            ):
                for generated_output in output.outputs:
                    if generated_output.text:
                        # Extract new tokens (not already seen)
                        new_text = generated_output.text
                        if new_text:
                            yield new_text
                            token_count += 1
                            
                            # Update active sequence tracking
                            self.active_sequences[request.request_id]['tokens_generated'] = token_count + 1
                            
                            # Check if generation is complete
                            if generated_output.finish_reason:
                                break
            
            # Update metrics
            decode_time = (time.time() - decode_start_time) * 1000
            total_time = (time.time() - start_time) * 1000
            
            self.requests_processed += 1
            self.total_decode_time += decode_time
            self.total_tokens_generated += token_count + 1  # Include prefill token
            
            # Clean up
            self.active_sequences[request.request_id]['status'] = 'completed'
            if request.request_id in self.injected_caches:
                del self.injected_caches[request.request_id]
            
            logging.info(f"Decode completed for {request.request_id}: "
                        f"tokens={token_count + 1}, decode_time={decode_time:.1f}ms, "
                        f"total_time={total_time:.1f}ms")
                        
        except Exception as e:
            self.active_sequences[request.request_id]['status'] = 'failed'
            logging.error(f"Decode generation failed for {request.request_id}: {e}")
            raise
    
    async def process_decode_request(self, decode_request: DecodeRequest) -> DecodeResult:
        """
        Process complete decode request and return final result.
        
        Alternative to streaming - returns complete generated text.
        """
        generated_tokens = []
        start_time = time.time()
        
        async for token in self.continue_generation(decode_request):
            generated_tokens.append(token)
        
        total_time = (time.time() - start_time) * 1000
        generated_text = "".join(generated_tokens)
        
        return DecodeResult(
            request_id=decode_request.inference_request.request_id,
            generated_text=generated_text,
            tokens_generated=len(generated_tokens),
            total_processing_time_ms=total_time,
            decode_time_ms=total_time - decode_request.prefill_result.processing_time_ms,
            cache_hit=True,
            finish_reason="length"  # or "stop"
        )
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get decode worker performance metrics."""
        avg_decode_time = (
            self.total_decode_time / self.requests_processed 
            if self.requests_processed > 0 else 0
        )
        
        avg_tokens_per_request = (
            self.total_tokens_generated / self.requests_processed
            if self.requests_processed > 0 else 0
        )
        
        tokens_per_second = (
            self.total_tokens_generated / (self.total_decode_time / 1000)
            if self.total_decode_time > 0 else 0
        )
        
        cache_hit_rate = (
            self.cache_hits / self.cache_injections
            if self.cache_injections > 0 else 0
        )
        
        return {
            'node_id': self.node_id,
            'worker_type': 'decode',
            'requests_processed': self.requests_processed,
            'active_sequences': len(self.active_sequences),
            'avg_decode_time_ms': avg_decode_time,
            'avg_tokens_per_request': avg_tokens_per_request,
            'tokens_per_second': tokens_per_second,
            'total_tokens_generated': self.total_tokens_generated,
            'cache_injections': self.cache_injections,
            'cache_hit_rate': cache_hit_rate,
            'engine_config': self.engine_config,
            'status': 'healthy'
        }
    
    # HTTP API Implementation
    async def handle_decode_request(self, request: web.Request) -> web.Response:
        """Handle HTTP decode request with KV cache."""
        try:
            data = await request.json()
            
            # Parse inference request
            inference_request = InferenceRequest(**data['inference_request'])
            
            # Parse prefill result
            prefill_data = data['prefill_result']
            prefill_result = PrefillResult(
                request_id=prefill_data['request_id'],
                first_token=prefill_data['first_token'],
                kv_cache_data=bytes.fromhex(prefill_data['kv_cache_data']),  # Decode from hex
                prompt_tokens=prefill_data['prompt_tokens'],
                cache_size_bytes=prefill_data['cache_size_bytes'],
                processing_time_ms=prefill_data['processing_time_ms'],
                cache_hash=prefill_data['cache_hash']
            )
            
            decode_request = DecodeRequest(
                inference_request=inference_request,
                prefill_result=prefill_result
            )
            
            # Process decode
            result = await self.process_decode_request(decode_request)
            
            return web.json_response({
                'request_id': result.request_id,
                'generated_text': result.generated_text,
                'tokens_generated': result.tokens_generated,
                'total_processing_time_ms': result.total_processing_time_ms,
                'decode_time_ms': result.decode_time_ms,
                'cache_hit': result.cache_hit,
                'finish_reason': result.finish_reason
            })
            
        except Exception as e:
            logging.error(f"Decode request failed: {e}")
            return web.json_response(
                {'error': str(e)}, 
                status=500
            )
    
    async def handle_streaming_decode(self, request: web.Request) -> web.StreamResponse:
        """Handle streaming decode request."""
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'application/x-ndjson'
        await response.prepare(request)
        
        try:
            data = await request.json()
            
            # Parse requests (same as above)
            inference_request = InferenceRequest(**data['inference_request'])
            prefill_data = data['prefill_result']
            prefill_result = PrefillResult(
                request_id=prefill_data['request_id'],
                first_token=prefill_data['first_token'],
                kv_cache_data=bytes.fromhex(prefill_data['kv_cache_data']),
                prompt_tokens=prefill_data['prompt_tokens'],
                cache_size_bytes=prefill_data['cache_size_bytes'],
                processing_time_ms=prefill_data['processing_time_ms'],
                cache_hash=prefill_data['cache_hash']
            )
            
            decode_request = DecodeRequest(
                inference_request=inference_request,
                prefill_result=prefill_result
            )
            
            # Stream tokens
            async for token in self.continue_generation(decode_request):
                chunk = {
                    'token': token,
                    'request_id': inference_request.request_id
                }
                
                await response.write(
                    (json.dumps(chunk) + '\n').encode('utf-8')
                )
                await response.drain()
            
            # Send completion marker
            await response.write(b'{"done": true}\n')
            
        except Exception as e:
            import json
            error_chunk = {'error': str(e)}
            await response.write(
                (json.dumps(error_chunk) + '\n').encode('utf-8')
            )
        
        return response
    
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
        self.app.router.add_post('/decode', self.handle_decode_request)
        self.app.router.add_post('/decode_stream', self.handle_streaming_decode)
        self.app.router.add_get('/health', self.handle_health_check)
        self.app.router.add_get('/metrics', self.handle_metrics)
        
        # Start server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        
        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()
        
        logging.info(f"DecodeWorker {self.node_id} started on {self.host}:{self.port}")
    
    async def stop_server(self) -> None:
        """Stop the HTTP server."""
        if self.site:
            await self.site.stop()
        if self.runner:
            await self.runner.cleanup()
        
        logging.info(f"DecodeWorker {self.node_id} stopped")


class MockDecodeEngine:
    """Mock implementation for testing without vLLM."""
    
    def __init__(self, model_name: str, **config):
        self.model_name = model_name
        self.config = config
        self.injected_caches: Dict[str, Any] = {}
        
    async def inject_kv_cache(self, request_id: str, kv_cache: Any) -> bool:
        """Mock cache injection."""
        self.injected_caches[request_id] = kv_cache
        return True
        
    async def generate(self, prompt: str, sampling_params, request_id: str, 
                      continue_from_cache: bool = False):
        """Mock generation with streaming."""
        if continue_from_cache and request_id in self.injected_caches:
            # Simulate faster generation with cache
            tokens = ["world", "!", "This", "is", "a", "test", "response"]
        else:
            tokens = ["Hello", "world", "without", "cache"]
        
        # Mock streaming output
        for i, token in enumerate(tokens):
            await asyncio.sleep(0.05)  # Simulate generation delay
            
            class MockOutput:
                def __init__(self, text: str, finish_reason: Optional[str] = None):
                    self.text = text
                    self.finish_reason = finish_reason
            
            class MockResult:
                def __init__(self, outputs: List[MockOutput]):
                    self.outputs = outputs
            
            finish_reason = "length" if i == len(tokens) - 1 else None
            yield MockResult([MockOutput(token, finish_reason)])


async def main():
    """Main entry point for running decode worker."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Decode Worker for Disaggregated Inference")
    parser.add_argument("--node-id", required=True, help="Unique node identifier")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument("--model", default="facebook/opt-125m", help="Model name")
    parser.add_argument("--max-seqs", type=int, default=64, help="Max concurrent sequences")
    parser.add_argument("--max-len", type=int, default=8192, help="Max sequence length")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and start worker
    worker = DecodeWorker(
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
        logging.info("Shutting down decode worker...")
    finally:
        await worker.stop_server()


if __name__ == "__main__":
    asyncio.run(main())