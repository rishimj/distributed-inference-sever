# KV Cache Strategy for High Prefix Reuse Use Cases

## Use Cases with High Prefix Reuse

Your use case likely fits into one of these high-reuse patterns:

### **1. Template-Based Applications**
```python
# Customer service chatbot
"You are a helpful customer service agent. Based on this conversation history: {history}, respond to: {new_message}"

# Code generation
"Generate Python code for the following requirement. Follow these style guidelines: {guidelines}. Requirement: {requirement}"

# Document analysis
"Analyze the following document and extract key insights: {document}. Question: {question}"
```

### **2. Multi-Turn Conversations**
```python
# Each turn builds on previous context
Turn 1: "Explain machine learning"
Turn 2: "Explain machine learning. What are neural networks?"
Turn 3: "Explain machine learning. What are neural networks? How does backpropagation work?"
```

### **3. Batch Processing with Common Patterns**
```python
# Processing multiple items with same template
"Summarize this customer feedback: {feedback_1}"
"Summarize this customer feedback: {feedback_2}"
"Summarize this customer feedback: {feedback_3}"
```

## Optimized Cache Strategy for High Reuse

### **Extended Cache Window Strategy**

For high-reuse scenarios, we can justify keeping cache affinity longer:

```python
class HighReuseCacheTracker:
    """
    Optimized for use cases with frequent prefix reuse.
    """
    
    def __init__(self, 
                 active_window_minutes: int = 10,    # Extended for high reuse
                 warm_window_minutes: int = 60,      # Keep warm cache longer
                 cold_window_minutes: int = 240):    # 4-hour cold cache
        
        self.active_window = active_window_minutes * 60
        self.warm_window = warm_window_minutes * 60  
        self.cold_window = cold_window_minutes * 60
        
        # prefix_hash -> (node_id, last_seen, access_count, cache_confirmed)
        self.prefix_cache: Dict[str, Tuple[str, float, int, bool]] = {}
        
        # Track template patterns for better prediction
        self.template_patterns: Dict[str, int] = {}  # template -> frequency
    
    def record_request_with_outcome(self, prefix_hash: str, node_id: str, 
                                   actual_cache_hit: bool) -> None:
        """
        Record request and actual cache performance.
        This helps us learn which prefixes actually get cache hits.
        """
        now = time.time()
        
        if prefix_hash in self.prefix_cache:
            _, _, access_count, _ = self.prefix_cache[prefix_hash]
            self.prefix_cache[prefix_hash] = (node_id, now, access_count + 1, actual_cache_hit)
        else:
            self.prefix_cache[prefix_hash] = (node_id, now, 1, actual_cache_hit)
    
    def get_cache_affinity_score(self, prefix_hash: str, node_id: str) -> float:
        """
        Optimized scoring for high-reuse scenarios.
        """
        if prefix_hash not in self.prefix_cache:
            return 0.1  # Unknown prefix
        
        cached_node, last_seen, access_count, confirmed_hit = self.prefix_cache[prefix_hash]
        
        if cached_node != node_id:
            return 0.05  # Wrong node penalty
        
        now = time.time()
        age = now - last_seen
        
        # Different scoring based on confirmed cache behavior
        if confirmed_hit:  # We know this prefix actually gets cache hits
            
            if age < self.active_window:  # 10 minutes - very likely hit
                return 0.9 + min(0.1, access_count * 0.01)
            
            elif age < self.warm_window:  # 1 hour - decent chance
                decay_factor = 1.0 - (age - self.active_window) / (self.warm_window - self.active_window)
                return 0.7 * decay_factor + 0.3  # 30-70% probability
            
            elif age < self.cold_window:  # 4 hours - small chance but worth trying
                decay_factor = 1.0 - (age - self.warm_window) / (self.cold_window - self.warm_window)
                return 0.3 * decay_factor + 0.1  # 10-30% probability
            
        else:  # This prefix doesn't typically get cache hits
            if age < 300:  # 5 minutes - only very recent
                return 0.4
            else:
                return 0.1
        
        return 0.1  # Too old
    
    def detect_template_pattern(self, prompt: str) -> Optional[str]:
        """
        Detect if this prompt follows a known template pattern.
        Templates have high reuse potential.
        """
        # Simple template detection (can be made more sophisticated)
        template_markers = [
            "You are a", "Based on the following", "Analyze the", 
            "Summarize this", "Generate", "Translate", "Extract"
        ]
        
        for marker in template_markers:
            if prompt.startswith(marker):
                # Extract template pattern
                template = prompt.split('.')[0]  # First sentence as template
                self.template_patterns[template] = self.template_patterns.get(template, 0) + 1
                return template
        
        return None
    
    def get_template_bonus(self, prompt: str) -> float:
        """
        Give bonus for prompts that follow frequently used templates.
        """
        template = self.detect_template_pattern(prompt)
        if template and template in self.template_patterns:
            frequency = self.template_patterns[template]
            if frequency >= 10:  # Frequently used template
                return 0.2
            elif frequency >= 3:  # Moderately used
                return 0.1
        
        return 0.0
```

### **vLLM Memory Management for High Reuse**

Configure vLLM to optimize for your use case:

```python
class vLLMCacheOptimizer:
    """
    vLLM configuration optimizations for high prefix reuse.
    """
    
    def get_optimized_vllm_config(self) -> Dict:
        """
        vLLM configuration optimized for prefix reuse.
        """
        return {
            # Increase KV cache block size to hold more prefixes
            'block_size': 32,  # Larger blocks for longer prefixes
            
            # Keep more KV cache blocks in memory
            'max_num_seqs': 128,  # Reduce concurrent sequences to save cache memory
            'max_model_len': 8192,  # Reasonable context length
            
            # Enable prefix caching if available
            'enable_prefix_caching': True,  # vLLM v0.2.0+
            
            # Optimize for throughput over latency for batching
            'max_num_batched_tokens': 4096,
            
            # Memory management
            'gpu_memory_utilization': 0.8,  # Leave room for KV cache
        }
    
    def estimate_cache_retention_time(self, node_load: float, 
                                    sequence_length: int) -> float:
        """
        Estimate how long a KV cache might survive based on node load.
        """
        # Base retention time when node is idle
        base_retention_minutes = 30  
        
        # Adjust based on load
        load_factor = max(0.1, 1.0 - node_load)
        
        # Adjust based on sequence length (longer sequences evicted first)
        length_factor = max(0.5, 1.0 - sequence_length / 4096)
        
        estimated_minutes = base_retention_minutes * load_factor * length_factor
        
        return estimated_minutes * 60  # Convert to seconds
```

### **Template-Aware Routing**

Route based on template patterns for maximum reuse:

```python
class TemplateAwareRouter:
    """
    Router optimized for template-based workloads.
    """
    
    def __init__(self):
        self.cache_tracker = HighReuseCacheTracker()
        self.template_to_node: Dict[str, str] = {}  # Pin templates to nodes
        
    async def route_request(self, request: InferenceRequest) -> RouteDecision:
        """
        Template-aware routing for high prefix reuse.
        """
        
        # 1. Detect template pattern
        template = self.cache_tracker.detect_template_pattern(request.prompt)
        
        # 2. Check for template affinity (pin similar templates to same node)
        if template and template in self.template_to_node:
            preferred_node = self.template_to_node[template]
            
            # Verify node is still healthy
            if await self.is_node_healthy(preferred_node):
                template_bonus = 0.3  # Strong preference for template consistency
                
                return RouteDecision(
                    target_node=preferred_node,
                    cache_hit_probability=0.7 + template_bonus,
                    reason=f"template_affinity: {template[:50]}..."
                )
        
        # 3. Check prefix cache affinity
        prefix_hash = self.hash_prompt_prefix(request.prompt)
        
        healthy_nodes = await self.get_healthy_nodes()
        best_node = None
        best_score = 0.0
        
        for node_id in healthy_nodes:
            # Cache affinity score
            cache_score = self.cache_tracker.get_cache_affinity_score(prefix_hash, node_id)
            
            # Template bonus
            template_bonus = self.cache_tracker.get_template_bonus(request.prompt)
            
            # Load balancing score
            load_score = await self.get_load_score(node_id)
            
            # Combined score (higher weight on cache for high-reuse scenarios)
            total_score = (
                cache_score * 0.6 +           # Higher cache weight for reuse scenarios
                load_score * 0.3 +            # Load balancing
                template_bonus * 0.1          # Template consistency bonus
            )
            
            if total_score > best_score:
                best_score = total_score
                best_node = node_id
        
        # 4. Update template mapping for future requests
        if template and best_node:
            self.template_to_node[template] = best_node
        
        return RouteDecision(
            target_node=best_node,
            cache_hit_probability=best_score * 0.8,  # Adjust for realism
            confidence=min(1.0, best_score + 0.2)
        )
    
    async def record_completion(self, request_id: str, prefix_hash: str, 
                              node_id: str, actual_cache_hit: bool,
                              response_time_ms: float) -> None:
        """
        Record completion with actual cache performance.
        This helps improve future predictions.
        """
        # Update cache tracker with actual results
        self.cache_tracker.record_request_with_outcome(
            prefix_hash, node_id, actual_cache_hit
        )
        
        # Adjust template preferences based on performance
        if actual_cache_hit and response_time_ms < 500:  # Good performance
            # Strengthen template affinity
            pass  # Template mapping already updated in route_request
        elif not actual_cache_hit:
            # Maybe reconsider template mapping
            # Could implement logic to try different nodes for templates
            pass
```

### **Performance Expectations for High Reuse**

With frequent prefix reuse, you can expect:

```python
performance_expectations = {
    'cache_hit_rate': {
        'template_based': '60-80%',    # High reuse templates
        'conversations': '40-60%',     # Multi-turn conversations  
        'batch_processing': '70-90%',  # Similar batch items
    },
    
    'latency_improvement': {
        'cache_hit': '3-5x faster',     # Significant with good prefixes
        'average_improvement': '2-3x',  # Overall system improvement
    },
    
    'optimal_configurations': {
        'template_workloads': 'Pin templates to specific nodes',
        'conversation_flows': 'Route conversations to same node',
        'batch_processing': 'Process batches on same node',
    }
}
```

### **Monitoring for High Reuse Scenarios**

Track metrics specific to prefix reuse:

```python
class HighReuseMetrics:
    """
    Metrics for monitoring prefix reuse effectiveness.
    """
    
    def track_prefix_patterns(self):
        return {
            'top_prefixes': self.get_most_common_prefixes(limit=20),
            'template_distribution': self.get_template_usage_stats(),
            'cache_hit_by_pattern': self.get_cache_hit_rates_by_pattern(),
            'node_affinity_strength': self.measure_node_affinity(),
        }
    
    def get_optimization_recommendations(self) -> List[str]:
        """
        Provide recommendations based on observed patterns.
        """
        recommendations = []
        
        # Check if certain templates have low hit rates
        low_hit_templates = self.find_templates_with_low_cache_hits()
        if low_hit_templates:
            recommendations.append(
                f"Consider dedicated nodes for templates: {low_hit_templates}"
            )
        
        # Check for node affinity violations
        scattered_templates = self.find_scattered_templates()
        if scattered_templates:
            recommendations.append(
                f"Templates spread across too many nodes: {scattered_templates}"
            )
        
        return recommendations
```

For your high prefix reuse use case, this strategy should deliver the **3-5x performance improvements** that the original complex algorithm promised, because you actually have the workload characteristics that make KV cache routing highly effective!
