# Benchmark Results

This directory contains benchmark results for ModernAraBERT.

## Directory Structure

```
results/
├── README.md (this file)
├── sa/
│   └── <model_name>_<dataset_name>_<timestamp>_results.json
├── ner/
│   └── <model_name>_<dataset_name>_<timestamp>_results.json
└── qa/
    └── <model_name>_<dataset_name>_<timestamp>_results.json
```

## Reproducing Results

To reproduce these results, follow the instructions in [BENCHMARKING.md](../docs/BENCHMARKING.md):

```bash
# Sentiment Analysis
bash scripts/benchmarking/run_sa_benchmark.sh --config configs/benchmarking/sa_benchmark.yaml --datasets hard

# Named Entity Recognition
bash scripts/benchmarking/run_ner_benchmark.sh --config configs/benchmarking/ner_benchmark.yaml

# Question Answering
bash scripts/benchmarking/run_qa_benchmark.sh --config configs/benchmarking/qa_benchmark.yaml
```
## Citation

If you use these results, please cite:

```bibtex
@inproceedings{<paper_id>,
  title={Efficient Adaptation of English Language Models for Low-Resource and Morphologically Rich Languages: The Case of Arabic},
  author={Maher, Eldamaty, Ashraf, ElShawi, Mostafa},
  booktitle={Proceedings of <conference_name>},
  year={2025}
}
```

## Notes

- All results use frozen encoder + task-specific head fine-tuning
- Random seed set to 42 for reproducibility
- Results may vary slightly due to hardware differences
- See paper for complete experimental setup details
