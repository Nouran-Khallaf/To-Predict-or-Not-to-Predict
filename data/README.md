# Data

This directory stores the data needed to run the uncertainty benchmark.

## Expected folders

```text
data/raw/          Original training data
data/processed/    Cleaned or transformed data
data/predictions/  Fold-level prediction CSV files
data/sample/       Small example files for testing
```

## Required training columns

For the current English readability setup:

- `Sentence`
- `Rating`

## Required prediction CSV columns

- `Sentence` or `text`
- `True Label` or `Label`
- `Lang`

## Label mapping

The current binary mapping is dataset-specific and will be implemented in:

```text
src/uncertainty_benchmark/data/label_mapping.py
```
