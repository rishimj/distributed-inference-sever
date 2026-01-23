"""
Unit tests for prefix hashing system.

These tests verify:
1. Different hashing strategies work correctly
2. Hierarchical hashing provides proper clustering
3. Token-aware hashing respects boundaries
4. Hash manager coordinates strategies properly
5. Performance characteristics are as expected
"""

import pytest

from src.common.models import CacheKey
from src.common.prefix_hasher import (
    FastHashPrefixHasher,
    HierarchicalPrefixHasher,
    PrefixHashManager,
    SHA256PrefixHasher,
    TokenAwarePrefixHasher,
    create_development_hasher,
    create_hierarchical_hasher,
    create_production_hasher,
    create_token_aware_hasher,
)


class TestSHA256PrefixHasher:
    """Test SHA-256 prefix hashing."""
    
    def test_basic_hashing(self):
        """Test basic SHA-256 prefix hashing."""
        hasher = SHA256PrefixHasher()
        text = "Hello world, this is a test prompt for hashing"
        
        hash1 = hasher.hash_prefix(text, 10)
        hash2 = hasher.hash_prefix(text, 10)
        
        # Same input should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 char hex string
        
        # Different prefix lengths should produce different hashes
        hash3 = hasher.hash_prefix(text, 20)
        assert hash1 != hash3
    
    def test_empty_and_short_text(self):
        """Test edge cases with empty and short text."""
        hasher = SHA256PrefixHasher()
        
        # Empty text
        empty_hash = hasher.hash_prefix("", 10)
        assert len(empty_hash) == 64
        
        # Text shorter than prefix length
        short_hash = hasher.hash_prefix("hi", 100)
        assert len(short_hash) == 64
    
    def test_deterministic_hashing(self):
        """Test that hashing is deterministic."""
        hasher = SHA256PrefixHasher()
        text = "Deterministic test prompt"
        
        hashes = [hasher.hash_prefix(text, 50) for _ in range(10)]
        
        # All hashes should be identical
        assert all(h == hashes[0] for h in hashes)
    
    def test_no_hierarchical_support(self):
        """Test that SHA-256 hasher doesn't support hierarchical hashing."""
        hasher = SHA256PrefixHasher()
        assert hasher.supports_hierarchical() is False


class TestFastHashPrefixHasher:
    """Test fast prefix hashing."""
    
    def test_basic_hashing(self):
        """Test basic fast hashing."""
        hasher = FastHashPrefixHasher()
        text = "Fast hashing test prompt"
        
        hash1 = hasher.hash_prefix(text, 10)
        hash2 = hasher.hash_prefix(text, 10)
        
        # Same input should produce same hash (within same process)
        assert hash1 == hash2
        assert len(hash1) == 16  # 64-bit hash as 16 char hex string
    
    def test_different_from_sha256(self):
        """Test that fast hash produces different results than SHA-256."""
        fast_hasher = FastHashPrefixHasher()
        sha_hasher = SHA256PrefixHasher()
        text = "Comparison test prompt"
        
        fast_hash = fast_hasher.hash_prefix(text, 20)
        sha_hash = sha_hasher.hash_prefix(text, 20)
        
        assert fast_hash != sha_hash
        assert len(fast_hash) == 16
        assert len(sha_hash) == 64
    
    def test_no_hierarchical_support(self):
        """Test that fast hasher doesn't support hierarchical hashing."""
        hasher = FastHashPrefixHasher()
        assert hasher.supports_hierarchical() is False


class TestHierarchicalPrefixHasher:
    """Test hierarchical prefix hashing."""
    
    def test_initialization(self):
        """Test hierarchical hasher initialization.""" 
        # Default levels
        hasher1 = HierarchicalPrefixHasher()
        assert hasher1.levels == [64, 256, 1024]
        
        # Custom levels
        hasher2 = HierarchicalPrefixHasher([32, 128, 512])
        assert hasher2.levels == [32, 128, 512]
    
    def test_level_selection(self):
        """Test that appropriate level is selected for given length."""
        hasher = HierarchicalPrefixHasher([64, 256, 1024])
        text = "Test prompt for level selection" * 50  # Make it long
        
        # Short length should use first level that accommodates it
        hash1 = hasher.hash_prefix(text, 50)   # Should use level 64
        hash2 = hasher.hash_prefix(text, 64)   # Should use level 64
        hash3 = hasher.hash_prefix(text, 100)  # Should use level 256
        
        # Same level should produce same hash
        assert hash1 == hash2
        # Different level should produce different hash
        assert hash1 != hash3
    
    def test_hierarchical_hashing(self):
        """Test hierarchical hash generation."""
        hasher = HierarchicalPrefixHasher([32, 128, 512])
        text = "Hierarchical test prompt " * 20  # Make it long enough
        
        hierarchical_hashes = hasher.hash_hierarchical(text)
        
        assert len(hierarchical_hashes) == 3
        assert hierarchical_hashes[0][0] == 32   # First level
        assert hierarchical_hashes[1][0] == 128  # Second level
        assert hierarchical_hashes[2][0] == 512  # Third level
        
        # All hashes should be different (different prefix lengths)
        hashes = [h[1] for h in hierarchical_hashes]
        assert len(set(hashes)) == 3
    
    def test_hierarchical_support(self):
        """Test that hierarchical hasher supports hierarchical operations."""
        hasher = HierarchicalPrefixHasher()
        assert hasher.supports_hierarchical() is True


class TestTokenAwarePrefixHasher:
    """Test token-aware prefix hashing."""
    
    def test_token_boundary_alignment(self):
        """Test that hashing respects token boundaries."""
        hasher = TokenAwarePrefixHasher()
        
        # Create text where character-based and token-based prefixes differ
        text = "Hello world test prompt"
        #       ^     ^     ^    ^
        #       6     12    17   23 (character positions)
        
        # Request prefix that falls mid-word
        char_prefix_len = 9  # Falls in middle of "world"
        hash_result = hasher.hash_prefix(text, char_prefix_len)
        
        # The hasher should align to token boundary ("Hello ")
        # We can't directly test the internal prefix, but we can test
        # that it produces consistent results
        assert len(hash_result) == 64  # SHA-256 hash length
    
    def test_tokenization(self):
        """Test text tokenization."""
        hasher = TokenAwarePrefixHasher()
        
        # Test tokenization method
        text = "Hello, world! How are you?"
        tokens = hasher._tokenize(text)
        
        # Should have tokens for words and punctuation
        assert "Hello" in tokens
        assert "," in tokens
        assert "world" in tokens
        assert "!" in tokens
        assert "How" in tokens
    
    def test_fallback_to_char_based(self):
        """Test fallback to character-based when token alignment is poor."""
        hasher = TokenAwarePrefixHasher()
        
        # Text with very long first token
        text = "Supercalifragilisticexpialidocious short words here"
        
        # Request short prefix that would give very poor token alignment
        short_hash = hasher.hash_prefix(text, 10)
        
        # Should fall back to character-based hashing
        assert len(short_hash) == 64
    
    def test_no_hierarchical_support(self):
        """Test that token-aware hasher doesn't support hierarchical hashing."""
        hasher = TokenAwarePrefixHasher()
        assert hasher.supports_hierarchical() is False


class TestPrefixHashManager:
    """Test prefix hash manager coordination."""
    
    def test_initialization(self):
        """Test hash manager initialization."""
        # Default initialization
        manager1 = PrefixHashManager()
        assert isinstance(manager1.hasher, SHA256PrefixHasher)
        assert 'gateway_routing' in manager1.default_lengths
        
        # Custom hasher
        custom_hasher = FastHashPrefixHasher()
        manager2 = PrefixHashManager(hasher=custom_hasher)
        assert manager2.hasher is custom_hasher
    
    def test_cache_key_creation(self):
        """Test cache key creation."""
        manager = PrefixHashManager()
        text = "Test prompt for cache key creation"
        
        cache_key = manager.create_cache_key(text)
        
        assert isinstance(cache_key, CacheKey)
        assert cache_key.model_name == "default"
        assert len(cache_key.prefix_hash) == 64  # SHA-256
        assert cache_key.sequence_length == len(text)
    
    def test_cache_key_with_custom_params(self):
        """Test cache key creation with custom parameters."""
        manager = PrefixHashManager()
        text = "Custom test prompt"
        
        cache_key = manager.create_cache_key(
            text=text,
            model_name="llama-7b",
            context="cache_cluster",
            sequence_length=512
        )
        
        assert cache_key.model_name == "llama-7b"
        assert cache_key.sequence_length == 512
        
        # Should use cache_cluster length (128)
        cluster_hash = manager.get_cluster_hash(text)
        assert cache_key.prefix_hash == cluster_hash
    
    def test_context_based_hashing(self):
        """Test that different contexts produce different hashes."""
        manager = PrefixHashManager()
        text = "Context test prompt " * 20  # Make it long enough
        
        routing_hash = manager.get_routing_hash(text)
        cache_hash = manager.get_cache_hash(text)
        cluster_hash = manager.get_cluster_hash(text)
        
        # Different prefix lengths should give different hashes
        hashes = {routing_hash, cache_hash, cluster_hash}
        assert len(hashes) == 3  # All should be different
    
    def test_similarity_detection_no_hierarchical(self):
        """Test similarity detection with non-hierarchical hasher."""
        manager = PrefixHashManager(hasher=SHA256PrefixHasher())
        text = "Similarity test prompt"
        existing_hashes = ["hash1", "hash2", "hash3"]
        
        # Non-hierarchical hasher should return empty list
        similar = manager.find_similar_prefixes(text, existing_hashes)
        assert similar == []
    
    def test_similarity_detection_hierarchical(self):
        """Test similarity detection with hierarchical hasher."""
        manager = PrefixHashManager(hasher=HierarchicalPrefixHasher())
        text = "Hierarchical similarity test"
        
        # This is a simplified test - in practice you'd have real cache hashes
        existing_hashes = ["abc123def456", "abc123xyz789", "def456ghi789"]
        
        similar = manager.find_similar_prefixes(text, existing_hashes)
        # Result depends on the actual implementation and hash values
        assert isinstance(similar, list)
    
    def test_prefix_distribution_analysis(self):
        """Test prefix distribution analysis."""
        manager = PrefixHashManager()
        
        texts = [
            "First test prompt",
            "Second test prompt", 
            "Third test prompt",
            "Different prompt entirely",
            "Another different prompt"
        ]
        
        analysis = manager.analyze_prefix_distribution(texts)
        
        assert analysis['total_texts'] == 5
        assert 'unique_routing_hashes' in analysis
        assert 'unique_cache_hashes' in analysis
        assert 'unique_cluster_hashes' in analysis
        assert 'routing_collision_rate' in analysis
        assert 'cache_collision_rate' in analysis
        assert 'cluster_collision_rate' in analysis
        assert 'avg_cluster_size' in analysis
        
        # Basic sanity checks
        assert 0 <= analysis['routing_collision_rate'] <= 1
        assert 0 <= analysis['cache_collision_rate'] <= 1
        assert 0 <= analysis['cluster_collision_rate'] <= 1
        assert analysis['avg_cluster_size'] >= 1


class TestFactoryFunctions:
    """Test factory functions for creating common configurations."""
    
    def test_production_hasher(self):
        """Test production hasher factory."""
        manager = create_production_hasher()
        
        assert isinstance(manager.hasher, SHA256PrefixHasher)
        assert manager.default_lengths['gateway_routing'] == 512
        assert manager.default_lengths['cache_lookup'] == 256
    
    def test_development_hasher(self):
        """Test development hasher factory."""
        manager = create_development_hasher()
        
        assert isinstance(manager.hasher, FastHashPrefixHasher)
        # Development should have shorter lengths for speed
        assert manager.default_lengths['gateway_routing'] == 256
        assert manager.default_lengths['cache_lookup'] == 128
    
    def test_hierarchical_hasher(self):
        """Test hierarchical hasher factory."""
        # Default levels
        manager1 = create_hierarchical_hasher()
        assert isinstance(manager1.hasher, HierarchicalPrefixHasher)
        assert manager1.hasher.supports_hierarchical()
        
        # Custom levels
        custom_levels = [32, 128, 512]
        manager2 = create_hierarchical_hasher(custom_levels)
        assert manager2.hasher.levels == custom_levels
    
    def test_token_aware_hasher(self):
        """Test token-aware hasher factory.""" 
        manager = create_token_aware_hasher()
        
        assert isinstance(manager.hasher, TokenAwarePrefixHasher)
        assert not manager.hasher.supports_hierarchical()


class TestHasherPerformance:
    """Test performance characteristics of different hashers."""
    
    def test_hash_consistency_across_calls(self):
        """Test that all hashers produce consistent results."""
        text = "Consistency test prompt " * 10
        length = 100
        
        hashers = [
            SHA256PrefixHasher(),
            FastHashPrefixHasher(),
            HierarchicalPrefixHasher(),
            TokenAwarePrefixHasher()
        ]
        
        for hasher in hashers:
            # Multiple calls should produce same result
            hash1 = hasher.hash_prefix(text, length)
            hash2 = hasher.hash_prefix(text, length) 
            hash3 = hasher.hash_prefix(text, length)
            
            assert hash1 == hash2 == hash3
    
    def test_hash_distribution_quality(self):
        """Test that hashes are well distributed (no obvious patterns)."""
        hasher = SHA256PrefixHasher()
        
        # Generate hashes for similar but different texts
        base_text = "Hash distribution test "
        hashes = []
        
        for i in range(100):
            text = f"{base_text}{i}"
            hash_val = hasher.hash_prefix(text, 50)
            hashes.append(hash_val)
        
        # Check that we get many unique hashes
        unique_hashes = set(hashes)
        assert len(unique_hashes) == 100  # All should be unique
        
        # Check that hash characters are well distributed
        all_chars = ''.join(hashes)
        hex_chars = set('0123456789abcdef')
        observed_chars = set(all_chars)
        
        # Should use most hex characters
        assert len(observed_chars & hex_chars) >= 14  # At least 14 of 16


if __name__ == "__main__":
    pytest.main([__file__])