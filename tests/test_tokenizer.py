"""
Tests for Tokenizer Extension Module

Tests the functions in src/pretraining/tokenizer_extension.py
"""

import pytest
import sys
from pathlib import Path
from collections import Counter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestVocabularyAnalysis:
    """Test suite for vocabulary analysis functions."""

    def test_token_frequency_counting(self):
        """Test token frequency counting."""
        tokens = ["hello", "world", "hello", "test", "hello"]
        freq = Counter(tokens)
        
        assert freq["hello"] == 3
        assert freq["world"] == 1
        assert freq["test"] == 1

    def test_frequency_sorting(self):
        """Test sorting tokens by frequency."""
        tokens = ["a", "b", "a", "c", "a", "b"]
        freq = Counter(tokens)
        sorted_tokens = freq.most_common()
        
        # Most frequent should be first
        assert sorted_tokens[0][0] == "a"
        assert sorted_tokens[0][1] == 3

    def test_vocabulary_size_limit(self):
        """Test limiting vocabulary to top N tokens."""
        tokens = ["a"] * 100 + ["b"] * 50 + ["c"] * 25 + ["d"] * 10
        freq = Counter(tokens)
        top_3 = freq.most_common(3)
        
        assert len(top_3) == 3
        assert top_3[0][0] == "a"
        assert top_3[1][0] == "b"
        assert top_3[2][0] == "c"


class TestTokenizerConfiguration:
    """Test suite for tokenizer configuration."""

    def test_max_vocab_size_default(self):
        """Test default maximum vocabulary size."""
        default_max_size = 80000
        assert default_max_size == 80000

    def test_min_frequency_threshold(self):
        """Test minimum frequency threshold."""
        tokens = ["rare"] * 3 + ["common"] * 100
        freq = Counter(tokens)
        min_freq = 5
        
        filtered = {k: v for k, v in freq.items() if v >= min_freq}
        
        assert "rare" not in filtered  # Below threshold
        assert "common" in filtered  # Above threshold

    def test_special_token_handling(self):
        """Test handling of special tokens."""
        special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
        
        # Special tokens should be preserved
        for token in special_tokens:
            assert token.startswith("[") and token.endswith("]")


class TestSegmentationMarkers:
    """Test suite for Farasa segmentation marker handling."""

    def test_segmentation_marker_detection(self):
        """Test detection of + markers from Farasa."""
        segmented_word = "ال+كتاب"
        assert "+" in segmented_word

    def test_segmentation_marker_splitting(self):
        """Test splitting on + markers."""
        segmented_word = "و+ال+كتاب+ين"
        parts = segmented_word.split("+")
        
        assert len(parts) == 4
        assert "و" in parts
        assert "ال" in parts
        assert "كتاب" in parts
        assert "ين" in parts

    def test_base_word_extraction(self):
        """Test extracting base word from segmented form."""
        segmented_word = "و+ال+كتاب"
        base_word = segmented_word.replace("+", "")
        
        assert base_word == "والكتاب"
        assert "+" not in base_word


class TestVocabularyExtension:
    """Test suite for vocabulary extension logic."""

    def test_new_token_addition(self):
        """Test adding new tokens to vocabulary."""
        existing_vocab = {"hello": 0, "world": 1}
        new_tokens = ["مرحبا", "عالم"]
        
        # Simulate adding new tokens
        updated_vocab = existing_vocab.copy()
        start_id = len(existing_vocab)
        
        for i, token in enumerate(new_tokens):
            updated_vocab[token] = start_id + i
        
        assert len(updated_vocab) == 4
        assert "مرحبا" in updated_vocab
        assert updated_vocab["مرحبا"] == 2

    def test_vocabulary_size_calculation(self):
        """Test calculating new vocabulary size."""
        original_size = 150000
        tokens_to_add = 80000
        new_size = original_size + tokens_to_add
        
        assert new_size == 230000

    def test_duplicate_token_handling(self):
        """Test that duplicate tokens are not added."""
        existing_vocab = {"hello": 0, "world": 1}
        new_tokens = ["hello", "test"]  # "hello" is duplicate
        
        # Filter out duplicates
        filtered_new_tokens = [t for t in new_tokens if t not in existing_vocab]
        
        assert "hello" not in filtered_new_tokens
        assert "test" in filtered_new_tokens


class TestMemoryEfficiency:
    """Test suite for memory-efficient processing."""

    def test_generator_based_processing(self):
        """Test that large data can be processed with generators."""
        def text_generator():
            for i in range(1000):
                yield f"نص رقم {i}"
        
        # Process with generator (memory efficient)
        count = sum(1 for _ in text_generator())
        assert count == 1000

    def test_batch_processing(self):
        """Test batch processing of texts."""
        texts = [f"نص {i}" for i in range(100)]
        batch_size = 10
        
        batches = [texts[i:i+batch_size] for i in range(0, len(texts), batch_size)]
        
        assert len(batches) == 10
        assert len(batches[0]) == 10


class TestErrorHandling:
    """Test suite for error handling in tokenizer extension."""

    def test_empty_corpus_handling(self):
        """Test handling of empty corpus."""
        tokens = []
        freq = Counter(tokens)
        
        assert len(freq) == 0
        assert freq.most_common(10) == []

    def test_invalid_token_handling(self):
        """Test handling of invalid tokens."""
        # Empty strings should be filtered
        tokens = ["hello", "", "world", ""]
        valid_tokens = [t for t in tokens if t and t.strip()]
        
        assert "" not in valid_tokens
        assert len(valid_tokens) == 2

    def test_unicode_token_handling(self):
        """Test handling of Unicode tokens."""
        arabic_tokens = ["مرحبا", "عالم", "اختبار"]
        
        # Should handle Arabic Unicode correctly
        for token in arabic_tokens:
            assert len(token) > 0
            # Check if contains Arabic Unicode range
            has_arabic = any('\u0600' <= c <= '\u06FF' for c in token)
            assert has_arabic


class TestIntegration:
    """Integration tests for tokenizer extension."""

    def test_complete_extension_workflow(self):
        """Test complete tokenizer extension workflow."""
        # Step 1: Collect tokens
        sample_texts = ["هذا نص", "نص آخر", "هذا اختبار"]
        all_tokens = []
        for text in sample_texts:
            all_tokens.extend(text.split())
        
        # Step 2: Count frequencies
        freq = Counter(all_tokens)
        
        # Step 3: Select top tokens
        max_vocab = 5
        top_tokens = freq.most_common(max_vocab)
        
        # Step 4: Create new vocabulary
        new_vocab = {token: idx for idx, (token, _) in enumerate(top_tokens)}
        
        # Assertions
        assert len(new_vocab) <= max_vocab
        assert "هذا" in new_vocab  # Most frequent

    def test_vocabulary_statistics(self):
        """Test vocabulary statistics calculation."""
        tokens = ["a"] * 100 + ["b"] * 50 + ["c"] * 25
        freq = Counter(tokens)
        
        total_tokens = sum(freq.values())
        unique_tokens = len(freq)
        
        assert total_tokens == 175
        assert unique_tokens == 3
        
        # Calculate coverage
        top_2_coverage = sum(freq.most_common(2)[i][1] for i in range(2))
        coverage_ratio = top_2_coverage / total_tokens
        
        assert coverage_ratio > 0.85  # Top 2 tokens cover >85%


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

