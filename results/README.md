# Benchmark Results

This directory contains benchmark results for ModernAraBERT as reported in our LREC 2026 paper.

## Directory Structure

```
results/
├── README.md (this file)
├── sentiment_analysis/
│   ├── modernarabert_hard.csv
│   ├── modernarabert_ajgt.csv
│   ├── modernarabert_labr.csv
│   ├── arabert_hard.csv
│   ├── mbert_hard.csv
│   └── comparison_table.csv
├── named_entity_recognition/
│   ├── modernarabert_anercorp.csv
│   ├── arabert_anercorp.csv
│   ├── mbert_anercorp.csv
│   └── comparison_table.csv
└── question_answering/
    ├── modernarabert_arcd.csv
    ├── arabert_arcd.csv
    ├── mbert_arcd.csv
    └── comparison_table.csv
```

## Paper Results (LREC 2026)

### Sentiment Analysis (Macro-F1 %)

| Model | AJGT | HARD | LABR |
|-------|------|------|------|
| **ModernAraBERT** | **70.5** | **89.4** | **56.5** |
| AraBERT v1 | 58.0 | 72.7 | 45.5 |
| mBERT | 61.5 | 71.7 | 45.5 |

### Named Entity Recognition (Micro F1 %)

| Model | ANERCorp |
|-------|----------|
| **mBERT** | **90.7** |
| **ModernAraBERT** | 82.1 |
| AraBERT v1 | 78.9 |

### Question Answering (%)

| Model | Exact Match | F1-Score | Sentence Match |
|-------|-------------|----------|----------------|
| **ModernAraBERT** | **18.73** | **47.18** | **76.66** |
| mBERT | 15.27 | 46.12 | 63.11 |
| AraBERT v1 | 13.26 | 40.82 | 71.47 |

## Result File Format

Each CSV file contains detailed results with the following columns:

### Sentiment Analysis CSV Format
```csv
model,dataset,split,accuracy,precision,recall,f1_macro,f1_weighted,timestamp
modernarabert,hard,test,0.894,0.891,0.897,0.894,0.893,2025-01-15T10:30:00
```

### NER CSV Format
```csv
model,dataset,split,accuracy,precision,recall,f1_micro,f1_macro,per_ppm,loc,org,misc,timestamp
modernarabert,anercorp,test,0.890,0.823,0.819,0.821,0.805,0.870,0.820,0.810,0.720,2025-01-15T11:00:00
```

### QA CSV Format
```csv
model,dataset,split,exact_match,f1_score,sentence_match,timestamp
modernarabert,arcd,test,18.73,47.18,76.66,2025-01-15T12:00:00
```

## Reproducing Results

To reproduce these results, follow the instructions in [BENCHMARKING.md](../docs/BENCHMARKING.md):

```bash
# Sentiment Analysis
bash scripts/benchmarking/run_sa_benchmark.sh --model-name modernarabert --dataset hard

# Named Entity Recognition
bash scripts/benchmarking/run_ner_benchmark.sh --model-name modernarabert

# Question Answering
python src/benchmarking/qa/qa_benchmark.py --model-name modernarabert
```

## Comparing Results

Use the comparison script to generate comparison tables:

```bash
python scripts/compare_results.py \
    --results-dir ./results \
    --output-file comparison_table.md
```

## Hardware Resource Usage

Benchmark results also include resource usage metrics:

| Benchmark | Model | Peak RAM (GB) | Peak VRAM (GB) | Throughput (samples/sec) |
|-----------|-------|---------------|----------------|--------------------------|
| NER | ModernAraBERT | 1.49 | 0.83 | 495.8 |
| NER | AraBERT | 1.53 | 0.52 | 925.4 |
| NER | mBERT | 1.60 | 0.68 | - |
| QA | ModernAraBERT | 1.39 | 3.22 | - |
| QA | AraBERT | 1.42 | 2.07 | - |
| QA | mBERT | 1.46 | 2.84 | - |
| SA | ModernAraBERT | 1.36 | 0.82 | - |
| SA | AraBERT | 1.65 | 0.52 | - |
| SA | mBERT | 1.66 | 0.68 | - |

## Visualization

Generate visualizations of results:

```python
python scripts/visualize_results.py \
    --results-dir ./results \
    --output-dir ./results/figures
```

This will create:
- Bar charts comparing models across tasks
- Performance vs. resource usage scatter plots
- Per-dataset breakdown visualizations

## Statistical Significance

Results include statistical significance tests (t-test, McNemar's test where applicable). See individual result files for p-values and confidence intervals.

## Citation

If you use these results, please cite:

```bibtex
@inproceedings{eldamaty2026modernarabert,
  title={Efficient Adaptation of English Language Models for Low-Resource and Morphologically Rich Languages: The Case of Arabic},
  author={Eldamaty, Ahmed and Maher, Mohamed and Mostafa, Mohamed and Ashraf, Mariam and ElShawi, Radwa},
  booktitle={Proceedings of LREC-COLING 2026},
  year={2026}
}
```

## Notes

- All results use frozen encoder + task-specific head fine-tuning
- Random seed set to 42 for reproducibility
- Results may vary slightly due to hardware differences
- See paper for complete experimental setup details

