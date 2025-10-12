"""
Tests for Data Preprocessing Module

Tests the functions in src/pretraining/data_preprocessing.py
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pretraining.data_preprocessing import (
    remove_diacritics,
    remove_tatweel,
    normalize_arabic_text,
    has_english_words,
    count_words,
)


class TestArabicNormalization:
    """Test suite for Arabic text normalization functions."""

    def test_remove_diacritics(self, sample_arabic_text_with_diacritics):
        """Test diacritics removal."""
        result = remove_diacritics(sample_arabic_text_with_diacritics)
        
        # Should not contain diacritics
        assert "َ" not in result  # Fatha
        assert "ُ" not in result  # Damma
        assert "ِ" not in result  # Kasra
        assert "ْ" not in result  # Sukun
        
        # Should still contain base letters
        assert "الحمد" in result
        assert "لله" in result

    def test_remove_tatweel(self, sample_arabic_text_with_tatweel):
        """Test tatweel (elongation) removal."""
        result = remove_tatweel(sample_arabic_text_with_tatweel)
        
        # Should not contain tatweel character
        assert "ـ" not in result
        
        # Should contain normalized words
        assert "السلام" in result
        assert "عليكم" in result

    def test_normalize_arabic_text_complete(self):
        """Test complete normalization pipeline."""
        text = "الحَمْدُ لِلَّهِ رَبِّ الْعَالَمِيـــــنَ"
        result = normalize_arabic_text(text)
        
        # Should be clean of diacritics and tatweel
        assert "َ" not in result
        assert "ُ" not in result
        assert "ِ" not in result
        assert "ـ" not in result
        
        # Should be properly normalized
        assert len(result) < len(text)

    def test_normalize_arabic_text_preserves_meaning(self, sample_arabic_text):
        """Test that normalization preserves base text."""
        result = normalize_arabic_text(sample_arabic_text)
        
        # Should preserve original words (no diacritics to remove)
        assert "نص" in result
        assert "عربي" in result
        assert "للاختبار" in result or "اختبار" in result


class TestTextFiltering:
    """Test suite for text filtering functions."""

    def test_has_english_words_pure_arabic(self, sample_arabic_text):
        """Test English detection with pure Arabic text."""
        assert has_english_words(sample_arabic_text) == False

    def test_has_english_words_mixed_text(self, sample_mixed_text):
        """Test English detection with mixed text."""
        assert has_english_words(sample_mixed_text) == True

    def test_has_english_words_english_only(self):
        """Test English detection with English-only text."""
        text = "This is English text only"
        assert has_english_words(text) == True

    def test_count_words_arabic(self, sample_arabic_text):
        """Test word counting for Arabic text."""
        count = count_words(sample_arabic_text)
        
        # Should count Arabic words correctly
        assert count > 0
        assert count < 20  # Reasonable range for sample

    def test_count_words_empty(self):
        """Test word counting for empty text."""
        count = count_words("")
        assert count == 0

    def test_count_words_whitespace(self):
        """Test word counting for whitespace only."""
        count = count_words("   \n\t  ")
        assert count == 0


class TestTextFileProcessing:
    """Test suite for text file processing."""

    def test_text_file_exists(self, sample_text_file):
        """Test that sample text file was created."""
        assert sample_text_file.exists()
        assert sample_text_file.is_file()

    def test_text_file_readable(self, sample_text_file):
        """Test that sample text file is readable."""
        content = sample_text_file.read_text(encoding="utf-8")
        assert len(content) > 0
        assert "عربي" in content

    def test_text_file_has_multiple_lines(self, sample_text_file):
        """Test that sample text file has multiple lines."""
        lines = sample_text_file.read_text(encoding="utf-8").split("\n")
        assert len(lines) >= 3


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    def test_normalize_empty_string(self):
        """Test normalization of empty string."""
        result = normalize_arabic_text("")
        assert result == ""

    def test_normalize_none_handling(self):
        """Test that None input is handled gracefully."""
        # Should either return empty string or raise appropriate error
        try:
            result = normalize_arabic_text(None)
            assert result == "" or result is None
        except (TypeError, AttributeError):
            # Expected behavior for None input
            pass

    def test_normalize_special_characters(self):
        """Test normalization with special characters."""
        text = "نص عربي 123 !@#$%"
        result = normalize_arabic_text(text)
        
        # Should preserve numbers and some special chars
        assert "نص" in result
        assert "عربي" in result

    def test_count_words_unicode(self):
        """Test word counting with various Unicode characters."""
        text = "نص عربي 中文 English"
        count = count_words(text)
        assert count > 0


class TestDataValidation:
    """Test suite for data validation functions."""

    def test_word_count_filter_min(self):
        """Test minimum word count filtering."""
        short_text = "نص قصير"
        count = count_words(short_text)
        
        # Should be less than typical minimum (100 words)
        assert count < 100

    def test_word_count_filter_max(self):
        """Test maximum word count filtering."""
        # Generate long text
        long_text = " ".join(["كلمة"] * 10000)
        count = count_words(long_text)
        
        # Should exceed typical maximum (8000 words)
        assert count > 8000

    def test_arabic_ratio_calculation(self, sample_mixed_text):
        """Test calculation of Arabic character ratio."""
        arabic_chars = sum(1 for c in sample_mixed_text if '\u0600' <= c <= '\u06FF')
        total_chars = len(sample_mixed_text.replace(" ", ""))
        
        if total_chars > 0:
            ratio = arabic_chars / total_chars
            # Mixed text should have ratio between 0 and 1
            assert 0 < ratio < 1


# Integration tests
class TestIntegration:
    """Integration tests for preprocessing pipeline."""

    def test_full_preprocessing_pipeline(self):
        """Test complete preprocessing pipeline."""
        text = "الحَمْدُ لِلَّهِ رَبِّ الْعَالَمِيـــــنَ with some English"
        
        # Step 1: Normalize
        normalized = normalize_arabic_text(text)
        
        # Step 2: Check for English
        has_english = has_english_words(normalized)
        
        # Step 3: Count words
        word_count = count_words(normalized)
        
        # Assertions
        assert len(normalized) < len(text)  # Normalization reduces length
        assert has_english == True  # Should detect English
        assert word_count > 0  # Should have words

    def test_preprocessing_preserves_structure(self, sample_text_file):
        """Test that preprocessing preserves document structure."""
        content = sample_text_file.read_text(encoding="utf-8")
        lines = content.split("\n")
        
        # Process each line
        processed_lines = [normalize_arabic_text(line) for line in lines]
        
        # Should have same number of lines
        assert len(processed_lines) == len(lines)
        
        # Each line should be processed
        for original, processed in zip(lines, processed_lines):
            if original.strip():  # Non-empty lines
                assert len(processed) <= len(original)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

