"""
Prefix hashing system for KV cache routing.

This module provides efficient prefix hashing for cache key generation and lookup.
Design decisions:
- Multiple hashing strategies for different use cases
- Configurable prefix lengths for different cache tiers
- Collision resistance with SHA-256 as default
- Support for hierarchical prefixes for cache clustering
"""

import hashlib
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

from .models import CacheKey


class PrefixHasher(ABC):
    """
    Abstract base class for prefix hashing strategies.
    
    Design choice: Strategy pattern for different hashing approaches.
    Tradeoff: More complexity vs flexibility for different use cases.
    """
    
    @abstractmethod
    def hash_prefix(self, text: str, length: int) -> str:
        """
        Hash a text prefix of given length.
        
        Args:
            text: Input text to hash
            length: Prefix length in characters
            
        Returns:
            Hex string hash of the prefix
        """
        pass
    
    @abstractmethod
    def supports_hierarchical(self) -> bool:
        """Return whether this hasher supports hierarchical prefixes."""
        pass


class SHA256PrefixHasher(PrefixHasher):
    """
    SHA-256 based prefix hasher with collision resistance.
    
    Best for: Production systems requiring security
    Tradeoff: Slower computation vs maximum collision resistance
    """
    
    def hash_prefix(self, text: str, length: int) -> str:
        """Hash prefix using SHA-256."""
        prefix = text[:length]
        return hashlib.sha256(prefix.encode('utf-8')).hexdigest()
    
    def supports_hierarchical(self) -> bool:
        """SHA-256 doesn't support hierarchical clustering naturally."""
        return False


class FastHashPrefixHasher(PrefixHasher):
    """
    Fast hash using Python's built-in hash function.
    
    Best for: Development and testing
    Tradeoff: Speed vs collision resistance and determinism across processes
    Warning: Python hash is not deterministic across process restarts!
    """
    
    def hash_prefix(self, text: str, length: int) -> str:
        """Hash prefix using Python's hash function."""
        prefix = text[:length]
        # Convert to positive integer and format as hex
        hash_val = hash(prefix) & 0x7FFFFFFFFFFFFFFF  # Remove sign bit
        return f"{hash_val:016x}"
    
    def supports_hierarchical(self) -> bool:
        """Fast hash doesn't support hierarchical clustering."""
        return False


class HierarchicalPrefixHasher(PrefixHasher):
    """
    Hierarchical prefix hasher that creates clustered cache keys.
    
    This hasher creates multiple hash levels to enable cache clustering:
    - Level 1: First 64 chars -> enables routing to cache clusters
    - Level 2: First 256 chars -> enables fine-grained cache lookup
    - Level 3: Full prefix -> exact cache match
    
    Best for: Large scale deployments with cache clustering
    Tradeoff: More complex routing vs better cache locality
    """
    
    def __init__(self, levels: List[int] = None):
        """
        Initialize hierarchical hasher.
        
        Args:
            levels: List of prefix lengths for each level
                   Default: [64, 256, 1024] 
        """
        self.levels = levels or [64, 256, 1024]
        self.base_hasher = SHA256PrefixHasher()
    
    def hash_prefix(self, text: str, length: int) -> str:
        """
        Hash prefix using the appropriate level.
        
        For hierarchical hashing, this returns the hash for the level
        that best matches the requested length.
        """
        # Find the appropriate level for this length
        appropriate_level = min(
            (level for level in self.levels if level >= length),
            default=self.levels[-1]
        )
        return self.base_hasher.hash_prefix(text, appropriate_level)
    
    def hash_hierarchical(self, text: str) -> List[Tuple[int, str]]:
        """
        Generate hierarchical hashes for all levels.
        
        Returns:
            List of (level, hash) tuples for cache clustering
        """
        return [
            (level, self.base_hasher.hash_prefix(text, level))
            for level in self.levels
        ]
    
    def supports_hierarchical(self) -> bool:
        """Hierarchical hasher supports hierarchical operations."""
        return True


class TokenAwarePrefixHasher(PrefixHasher):
    """
    Token-aware prefix hasher that respects token boundaries.
    
    This hasher tries to break prefixes at token boundaries rather than
    character boundaries for better semantic cache hits.
    
    Best for: LLM workloads where token alignment matters
    Tradeoff: More complex logic vs better semantic cache hits
    """
    
    def __init__(self, token_pattern: str = r'\w+|\W'):
        """
        Initialize token-aware hasher.
        
        Args:
            token_pattern: Regex pattern for tokenization
        """
        self.token_pattern = re.compile(token_pattern)
        self.base_hasher = SHA256PrefixHasher()
    
    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text using the configured pattern."""
        return self.token_pattern.findall(text)
    
    def hash_prefix(self, text: str, length: int) -> str:
        """
        Hash prefix respecting token boundaries.
        
        Algorithm:
        1. Tokenize the text
        2. Build prefix token by token until we exceed length
        3. Use the longest prefix that doesn't exceed length
        4. Hash the resulting token-aligned prefix
        """
        if length >= len(text):
            return self.base_hasher.hash_prefix(text, length)
        
        tokens = self._tokenize(text)
        prefix_chars = 0
        token_prefix = ""
        
        for token in tokens:
            if prefix_chars + len(token) > length:
                break
            token_prefix += token
            prefix_chars += len(token)
        
        # If token alignment gives us a very short prefix, fall back to char-based
        if prefix_chars < length * 0.8:  # Allow 20% shorter for token alignment
            return self.base_hasher.hash_prefix(text, length)
        
        return self.base_hasher.hash_prefix(token_prefix, len(token_prefix))
    
    def supports_hierarchical(self) -> bool:
        """Token-aware hasher doesn't support hierarchical clustering."""
        return False


class PrefixHashManager:
    """
    Manager for coordinating prefix hashing across the system.
    
    This class provides:
    - Consistent hashing strategies across services
    - Cache key generation with metadata
    - Prefix similarity detection for cache clustering
    
    Design choice: Centralized hashing coordination
    Tradeoff: Single point of configuration vs distributed complexity
    """
    
    def __init__(self, 
                 hasher: Optional[PrefixHasher] = None,
                 default_lengths: Optional[Dict[str, int]] = None):
        """
        Initialize prefix hash manager.
        
        Args:
            hasher: Hashing strategy to use (default: SHA256PrefixHasher)
            default_lengths: Default prefix lengths by context
        """
        self.hasher = hasher or SHA256PrefixHasher()
        self.default_lengths = default_lengths or {
            'gateway_routing': 512,   # For routing decisions
            'cache_lookup': 256,      # For exact cache lookup
            'cache_cluster': 128,     # For cache clustering
            'similarity': 64          # For similarity detection
        }
    
    def create_cache_key(self, 
                        text: str, 
                        model_name: str = "default",
                        context: str = "cache_lookup",
                        sequence_length: Optional[int] = None) -> CacheKey:
        """
        Create a cache key for the given text and context.
        
        Args:
            text: Input text to create key for
            model_name: Model name for multi-model deployments
            context: Usage context for length selection
            sequence_length: Optional override for sequence length
            
        Returns:
            CacheKey object for cache operations
        """
        prefix_length = self.default_lengths.get(context, 256)
        prefix_hash = self.hasher.hash_prefix(text, prefix_length)
        
        return CacheKey(
            prefix_hash=prefix_hash,
            model_name=model_name,
            sequence_length=sequence_length or len(text)
        )
    
    def find_similar_prefixes(self, 
                            text: str, 
                            existing_hashes: List[str],
                            similarity_threshold: float = 0.8) -> List[str]:
        """
        Find existing cache entries with similar prefixes.
        
        Args:
            text: Input text to find similarities for
            existing_hashes: List of existing prefix hashes
            similarity_threshold: Minimum similarity score (0.0-1.0)
            
        Returns:
            List of similar cache key hashes
        """
        if not self.hasher.supports_hierarchical():
            # For non-hierarchical hashers, we can't do similarity matching
            return []
        
        # For hierarchical hashers, we can use shorter prefixes for clustering
        if isinstance(self.hasher, HierarchicalPrefixHasher):
            # Use the shortest level for similarity clustering
            cluster_length = self.hasher.levels[0]
            cluster_hash = self.hasher.hash_prefix(text, cluster_length)
            
            # Find hashes that share the same cluster
            similar = []
            for existing_hash in existing_hashes:
                # This is a simplified approach - in practice you'd store
                # hierarchical hashes and compare cluster-level hashes
                if self._shares_cluster(cluster_hash, existing_hash):
                    similar.append(existing_hash)
            
            return similar
        
        return []
    
    def _shares_cluster(self, hash1: str, hash2: str) -> bool:
        """
        Check if two hashes share a cluster (simplified implementation).
        
        In a real implementation, this would compare stored hierarchical
        hash information rather than just checking hash prefixes.
        """
        # Simplified: just check if they share a prefix
        # Real implementation would use stored cluster information
        return hash1[:16] == hash2[:16]  # Compare first 16 chars
    
    def get_routing_hash(self, text: str) -> str:
        """Get hash for routing decisions."""
        length = self.default_lengths['gateway_routing']
        return self.hasher.hash_prefix(text, length)
    
    def get_cache_hash(self, text: str) -> str:
        """Get hash for cache lookup."""
        length = self.default_lengths['cache_lookup']
        return self.hasher.hash_prefix(text, length)
    
    def get_cluster_hash(self, text: str) -> str:
        """Get hash for cache clustering.""" 
        length = self.default_lengths['cache_cluster']
        return self.hasher.hash_prefix(text, length)
    
    def analyze_prefix_distribution(self, texts: List[str]) -> Dict:
        """
        Analyze prefix distribution for cache optimization.
        
        Args:
            texts: List of input texts to analyze
            
        Returns:
            Dictionary with distribution statistics
        """
        routing_hashes = [self.get_routing_hash(text) for text in texts]
        cache_hashes = [self.get_cache_hash(text) for text in texts]
        cluster_hashes = [self.get_cluster_hash(text) for text in texts]
        
        return {
            'total_texts': len(texts),
            'unique_routing_hashes': len(set(routing_hashes)),
            'unique_cache_hashes': len(set(cache_hashes)),
            'unique_cluster_hashes': len(set(cluster_hashes)),
            'routing_collision_rate': 1 - len(set(routing_hashes)) / len(texts),
            'cache_collision_rate': 1 - len(set(cache_hashes)) / len(texts),
            'cluster_collision_rate': 1 - len(set(cluster_hashes)) / len(texts),
            'avg_cluster_size': len(texts) / len(set(cluster_hashes)) if cluster_hashes else 0
        }


# Factory functions for common configurations

def create_production_hasher() -> PrefixHashManager:
    """Create a production-ready hasher with security focus."""
    return PrefixHashManager(
        hasher=SHA256PrefixHasher(),
        default_lengths={
            'gateway_routing': 512,
            'cache_lookup': 256, 
            'cache_cluster': 128,
            'similarity': 64
        }
    )


def create_development_hasher() -> PrefixHashManager:
    """Create a fast hasher for development and testing."""
    return PrefixHashManager(
        hasher=FastHashPrefixHasher(),
        default_lengths={
            'gateway_routing': 256,
            'cache_lookup': 128,
            'cache_cluster': 64,
            'similarity': 32
        }
    )


def create_hierarchical_hasher(levels: List[int] = None) -> PrefixHashManager:
    """Create a hierarchical hasher for large-scale deployments."""
    return PrefixHashManager(
        hasher=HierarchicalPrefixHasher(levels),
        default_lengths={
            'gateway_routing': 512,
            'cache_lookup': 256,
            'cache_cluster': 128,
            'similarity': 64
        }
    )


def create_token_aware_hasher(token_pattern: str = r'\w+|\W') -> PrefixHashManager:
    """Create a token-aware hasher for better semantic alignment."""
    return PrefixHashManager(
        hasher=TokenAwarePrefixHasher(token_pattern),
        default_lengths={
            'gateway_routing': 512,
            'cache_lookup': 256,
            'cache_cluster': 128,  
            'similarity': 64
        }
    )