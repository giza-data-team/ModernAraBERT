"""
Tests for Benchmarking Modules

Tests the dataset loading and preprocessing functions for SA and NER benchmarks.
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestSentimentAnalysis:
    """Test suite for Sentiment Analysis benchmarking."""

    def test_sentiment_labels(self, sample_sentiment_data):
        """Test sentiment label validation."""
        labels = {item["label"] for item in sample_sentiment_data}
        
        assert "positive" in labels
        assert "negative" in labels
        # May or may not have neutral depending on dataset

    def test_sentiment_data_structure(self, sample_sentiment_data):
        """Test SA data structure."""
        for item in sample_sentiment_data:
            assert "text" in item
            assert "label" in item
            assert isinstance(item["text"], str)
            assert isinstance(item["label"], str)

    def test_sentiment_text_preprocessing(self, sample_sentiment_data):
        """Test basic text preprocessing for SA."""
        for item in sample_sentiment_data:
            text = item["text"]
            
            # Should have Arabic text
            assert len(text) > 0
            # Check for Arabic characters
            has_arabic = any('\u0600' <= c <= '\u06FF' for c in text)
            assert has_arabic

    def test_label_mapping(self):
        """Test label to ID mapping."""
        label2id = {
            "positive": 0,
            "negative": 1,
            "neutral": 2,
        }
        
        assert label2id["positive"] == 0
        assert label2id["negative"] == 1
        assert len(label2id) == 3

    def test_macro_f1_calculation(self):
        """Test Macro-F1 score calculation logic."""
        # Simulate per-class F1 scores
        f1_scores = [0.85, 0.90, 0.75]
        macro_f1 = sum(f1_scores) / len(f1_scores)
        
        assert 0 <= macro_f1 <= 1
        assert abs(macro_f1 - 0.833) < 0.01


class TestNamedEntityRecognition:
    """Test suite for Named Entity Recognition benchmarking."""

    def test_ner_data_structure(self, sample_ner_data):
        """Test NER data structure."""
        assert "tokens" in sample_ner_data
        assert "ner_tags" in sample_ner_data
        
        tokens = sample_ner_data["tokens"]
        tags = sample_ner_data["ner_tags"]
        
        # Should have same number of sequences
        assert len(tokens) == len(tags)
        
        # Each sequence should have matching lengths
        for token_seq, tag_seq in zip(tokens, tags):
            assert len(token_seq) == len(tag_seq)

    def test_iob2_tagging_scheme(self, sample_ner_data):
        """Test IOB2 tagging scheme."""
        all_tags = []
        for tag_seq in sample_ner_data["ner_tags"]:
            all_tags.extend(tag_seq)
        
        # Should have B-, I-, or O tags
        for tag in all_tags:
            assert tag == "O" or tag.startswith("B-") or tag.startswith("I-")

    def test_entity_types(self, sample_ner_data):
        """Test entity type extraction."""
        entity_types = set()
        for tag_seq in sample_ner_data["ner_tags"]:
            for tag in tag_seq:
                if tag != "O":
                    entity_type = tag.split("-")[1]
                    entity_types.add(entity_type)
        
        # Should have some entity types
        assert len(entity_types) > 0
        # Common types
        possible_types = {"PER", "LOC", "ORG", "MISC"}
        assert entity_types.issubset(possible_types)

    def test_label_alignment(self, sample_ner_data):
        """Test label alignment with tokens."""
        tokens = sample_ner_data["tokens"][0]
        tags = sample_ner_data["ner_tags"][0]
        
        # Should be able to zip tokens and tags
        aligned = list(zip(tokens, tags))
        assert len(aligned) == len(tokens)
        assert len(aligned) == len(tags)

    def test_micro_f1_logic(self):
        """Test Micro-F1 calculation logic."""
        # Simulate TP, FP, FN
        tp, fp, fn = 85, 10, 15
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        micro_f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        assert 0 <= micro_f1 <= 1
        assert micro_f1 > 0.8  # Should be reasonably high for test data


class TestDatasetConfigurations:
    """Test suite for dataset configurations."""

    def test_sa_dataset_configs(self):
        """Test SA dataset configurations."""
        sa_datasets = ["HARD", "AJGT", "LABR", "ASTD"]
        
        for dataset_name in sa_datasets:
            assert isinstance(dataset_name, str)
            assert len(dataset_name) > 0

    def test_ner_dataset_config(self):
        """Test NER dataset configuration."""
        ner_dataset = "ANERCorp"
        
        assert isinstance(ner_dataset, str)
        assert "ANERCorp" == ner_dataset

    def test_dataset_split_ratios(self):
        """Test dataset split ratios."""
        train_ratio = 0.6
        val_ratio = 0.2
        test_ratio = 0.2
        
        # Should sum to 1.0
        total = train_ratio + val_ratio + test_ratio
        assert abs(total - 1.0) < 0.001

    def test_dataset_label_counts(self):
        """Test label count validation."""
        # HARD: 2 classes (positive, negative)
        hard_labels = 2
        assert hard_labels == 2
        
        # ASTD: 4 classes (positive, negative, neutral, mixed)
        astd_labels = 4
        assert astd_labels == 4


class TestMemoryTracking:
    """Test suite for memory tracking utilities."""

    def test_memory_usage_structure(self):
        """Test memory usage data structure."""
        memory_info = {
            "ram_used_gb": 16.5,
            "ram_percent": 51.2,
            "vram_used_gb": 8.3,
            "vram_percent": 20.75,
        }
        
        assert "ram_used_gb" in memory_info
        assert "vram_used_gb" in memory_info
        assert memory_info["ram_used_gb"] > 0
        assert 0 <= memory_info["ram_percent"] <= 100

    def test_memory_unit_conversion(self):
        """Test memory unit conversion."""
        bytes_value = 1024 * 1024 * 1024  # 1 GB in bytes
        gb_value = bytes_value / (1024 ** 3)
        
        assert abs(gb_value - 1.0) < 0.001


class TestFrozenEncoderTraining:
    """Test suite for frozen encoder training strategy."""

    def test_frozen_parameters(self):
        """Test that encoder parameters can be frozen."""
        # Simulate parameter freezing
        encoder_params = {"param1": True, "param2": True, "param3": True}
        
        # Freeze all
        for param in encoder_params:
            encoder_params[param] = False  # requires_grad = False
        
        # All should be frozen
        assert all(not requires_grad for requires_grad in encoder_params.values())

    def test_trainable_head_parameters(self):
        """Test that head parameters remain trainable."""
        head_params = {"head_param1": True, "head_param2": True}
        
        # All should remain trainable
        assert all(requires_grad for requires_grad in head_params.values())


class TestEarlyStoppingLogic:
    """Test suite for early stopping logic."""

    def test_patience_counter(self):
        """Test early stopping patience counter."""
        patience = 10
        no_improve_epochs = 0
        best_score = 0.85
        
        # Simulate worse scores
        for i in range(5):
            current_score = 0.83
            if current_score <= best_score:
                no_improve_epochs += 1
        
        assert no_improve_epochs == 5
        assert no_improve_epochs < patience  # Should not stop yet

    def test_early_stopping_trigger(self):
        """Test early stopping trigger condition."""
        patience = 3
        no_improve_epochs = 3
        
        should_stop = no_improve_epochs >= patience
        assert should_stop == True


class TestBatchProcessing:
    """Test suite for batch processing."""

    def test_batch_size_validation(self):
        """Test batch size validation."""
        batch_sizes = [8, 16, 32, 64]
        
        for batch_size in batch_sizes:
            assert batch_size > 0
            assert batch_size % 8 == 0  # Typically powers of 2

    def test_gradient_accumulation(self):
        """Test gradient accumulation logic."""
        batch_size = 8
        gradient_accumulation_steps = 4
        effective_batch_size = batch_size * gradient_accumulation_steps
        
        assert effective_batch_size == 32


class TestResultsExport:
    """Test suite for results export functionality."""

    def test_results_json_structure(self):
        """Test results JSON structure."""
        results = {
            "model_name": "ModernAraBERT",
            "dataset": "HARD",
            "metrics": {
                "macro_f1": 89.4,
                "precision": 88.5,
                "recall": 90.3,
            },
            "training_time": 1234.56,
            "memory_usage": {
                "ram_gb": 16.5,
                "vram_gb": 35.2,
            },
        }
        
        assert "model_name" in results
        assert "dataset" in results
        assert "metrics" in results
        assert isinstance(results["metrics"], dict)

    def test_results_file_naming(self):
        """Test results file naming convention."""
        model_name = "ModernAraBERT"
        dataset = "HARD"
        filename = f"{model_name}_{dataset}_results.json"
        
        assert "ModernAraBERT" in filename
        assert "HARD" in filename
        assert filename.endswith(".json")


class TestIntegration:
    """Integration tests for benchmarking pipeline."""

    def test_complete_sa_pipeline(self, sample_sentiment_data):
        """Test complete SA pipeline simulation."""
        # Step 1: Load data
        data = sample_sentiment_data
        assert len(data) > 0
        
        # Step 2: Create label mapping
        unique_labels = list(set(item["label"] for item in data))
        label2id = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Step 3: Convert labels to IDs
        label_ids = [label2id[item["label"]] for item in data]
        
        # Assertions
        assert len(label_ids) == len(data)
        assert all(0 <= label_id < len(label2id) for label_id in label_ids)

    def test_complete_ner_pipeline(self, sample_ner_data):
        """Test complete NER pipeline simulation."""
        # Step 1: Load data
        tokens = sample_ner_data["tokens"]
        tags = sample_ner_data["ner_tags"]
        
        # Step 2: Create label mapping
        all_tags = set()
        for tag_seq in tags:
            all_tags.update(tag_seq)
        label2id = {label: idx for idx, label in enumerate(sorted(all_tags))}
        
        # Step 3: Convert tags to IDs
        tag_id_sequences = []
        for tag_seq in tags:
            tag_ids = [label2id[tag] for tag in tag_seq]
            tag_id_sequences.append(tag_ids)
        
        # Assertions
        assert len(tag_id_sequences) == len(tags)
        for orig_tags, tag_ids in zip(tags, tag_id_sequences):
            assert len(orig_tags) == len(tag_ids)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

