# Quickstart

## 1. Install

```bash
pip install -e .
```

## 2. Add data

Put training data in:

```text
data/raw/
```

Put fold prediction CSVs in:

```text
data/predictions/
```

## 3. Edit config

Open:

```text
configs/english_mbert.yaml
```

Check:

```yaml
data:
  train_file: "./data/raw/readme_en_combined_all.xlsx"
  pred_csv_template: "./data/predictions/bert-base-multilingual-cased_fold_{fold_id}_val_predictions.csv"
  lang_name: "readme_en_combined_all"
```

## 4. Run one fold

```bash
python scripts/run_single_fold.py \
  --config configs/english_mbert.yaml \
  --fold 0
```

## 5. Run all configured folds

```bash
python scripts/run_folds.py \
  --config configs/english_mbert.yaml
```

## 6. Make paper tables

```bash
python scripts/make_tables.py \
  --results-dir results/english_validation_mbert
```

## 7. Make figures

```bash
python scripts/make_figures.py \
  --results-dir results/english_validation_mbert \
  --formats png pdf
```
