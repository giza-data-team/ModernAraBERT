"""
Pytest Configuration and Shared Fixtures

This file provides common fixtures used across all test modules.
"""

import pytest
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)


@pytest.fixture
def sample_arabic_text():
    """Sample Arabic text for testing."""
    return "هذا نص عربي للاختبار يحتوي على كلمات مختلفة"


@pytest.fixture
def sample_arabic_text_with_diacritics():
    """Sample Arabic text with diacritics."""
    return "الحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ"


@pytest.fixture
def sample_arabic_text_with_tatweel():
    """Sample Arabic text with tatweel (elongation)."""
    return "السلامـــــــ عليكمـــــــ"


@pytest.fixture
def sample_mixed_text():
    """Sample text with Arabic and English mixed."""
    return "هذا نص عربي with English words مختلط"


@pytest.fixture
def sample_text_file(temp_dir):
    """Create a sample text file."""
    file_path = temp_dir / "sample.txt"
    file_path.write_text("هذا نص عربي في ملف\nسطر ثاني\nسطر ثالث", encoding="utf-8")
    return file_path


@pytest.fixture
def sample_sentiment_data():
    """Sample sentiment analysis data."""
    return [
        {"text": "هذا منتج رائع جداً", "label": "positive"},
        {"text": "تجربة سيئة للغاية", "label": "negative"},
        {"text": "جيد ولكن يحتاج تحسين", "label": "neutral"},
    ]


@pytest.fixture
def sample_ner_data():
    """Sample NER data in IOB2 format."""
    return {
        "tokens": [
            ["محمد", "يعمل", "في", "القاهرة"],
            ["زار", "أحمد", "مصر"],
        ],
        "ner_tags": [
            ["B-PER", "O", "O", "B-LOC"],
            ["O", "B-PER", "B-LOC"],
        ],
    }

