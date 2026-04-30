# Uncertainty Benchmark

A small benchmark for evaluating uncertainty estimation methods for text classification and readability prediction.

The repository supports two main workflows:

1. **Evaluate saved prediction CSVs**  
   Use this when you already have fold-level prediction files with labels, probabilities, and optionally saved uncertainty scores such as `SMP`.

2. **Generate reporting outputs**  
   Use this after evaluation to create paper-ready CSV files, LaTeX tables, and figures.

All uncertainty scores should follow one convention:

```text
larger score = more uncertain
```

---

## 1. Repository structure

```text

├── configs/
│   ├── evaluate_saved_model_uncertainty.yaml
│   └── reporting_english_validation_mbert.yaml
├── data/
├── results/
├── scripts/
│   ├── evaluate_saved_model_uncertainty.py
│   ├── make_report_outputs.py
│   ├── run_folds.py
│   └── run_single_fold.py
├── src/uncertainty_benchmark/
│   ├── data/
│   ├── models/
│   ├── methods/
│   ├── metrics/
│   ├── reporting/
│   └── visualisation/
└── tests/
```

---

## 2. Installation

From the repository root:

```bash
conda create -n uncertainty-benchmark python=3.10 -y
conda activate uncertainty-benchmark
```

Install requirements:

```bash
pip install -r requirements.txt
```

Install the local `src` package:

```bash
pip install -e .
```

If imports fail on a cluster, add `src` to `PYTHONPATH`:

```bash
export PYTHONPATH=$PWD/src:$PYTHONPATH
```

Check the installation:

```bash
python - <<'PY'
import uncertainty_benchmark
print("uncertainty_benchmark import OK")
PY
```

---

## 3. Supported uncertainty methods

| ID | Method | Family |
|---|---|---|
| `SR` | Softmax Response | Deterministic |
| `ENT` | Predictive Entropy | Deterministic |
| `MARGIN` | Probability Margin | Deterministic |
| `SMP` | Sampled Max Probability | MC Dropout |
| `PV` | Probability Variance | MC Dropout |
| `BALD` | Bayesian Active Learning by Disagreement | MC Dropout |
| `ENT_MC` | MC Predictive Entropy | MC Dropout |
| `MD` | Mahalanobis Distance | Distance |
| `HUQ-MD` | Hybrid Uncertainty Quantification with MD | Hybrid |
| `LOF` | Local Outlier Factor | Outlier |
| `ISOF` | Isolation Forest | Outlier |

---

## 4. Evaluation from saved prediction CSVs

Run:

```bash
python scripts/evaluate_saved_model_uncertainty.py \
  --config configs/evaluate_saved_model_uncertainty.yaml
```

This is the main workflow when the model predictions already exist.

### Where to change file names

Edit this config file:

```text
configs/evaluate_saved_model_uncertainty.yaml
```

Change these fields when moving to a new experiment:

| What to change | Config location | Example |
|---|---|---|
| prediction CSV file pattern | `data.pred_csv_template` | `results/my_experiment/fold_{fold_id}/predictions.csv` |
| output folder | `outputs.outdir` | `results/my_experiment/metrics` |
| folds to evaluate | `folds.fold_ids` or `folds.n_folds` | `[0, 1, 2, 3, 4]` |
| text column names | `data.eval_text_column_candidates` | `[Sentence, text]` |
| gold label column names | `data.eval_true_label_column_candidates` | `[True Label, Label, true_label]` |
| predicted label column names | `data.eval_pred_label_column_candidates` | `[Pred Label, Prediction, pred_label]` |
| probability column pattern | `labels.probability_column_template` | `Prob_{class}` |
| class labels | `labels.classes` | `[simple, complex]` |
| methods to evaluate | `methods.enabled` | `[SR, SMP, ENT]` |
| saved score column names | `methods.saved_score_columns` | map `SMP` to the actual SMP column |
| score direction | `methods.score_direction` | `uncertainty` or `confidence` |
| ECE bins | `metrics.ece_bins` | `15` |
| TI fixed coverage | `metrics.ti_fixed_coverage` | `0.95` |

### Prediction CSV requirements

Each fold prediction CSV should contain:

| Column type | Examples |
|---|---|
| text | `Sentence`, `text` |
| gold label | `True Label`, `Label`, `true_label` |
| predicted label | `Pred Label`, `Prediction`, `pred_label` |
| class probabilities | `Prob_simple`, `Prob_complex` |
| optional saved uncertainty scores | `SMP`, `PV`, `BALD`, `ENT_MC`, etc. |

`SR`, `ENT`, and `MARGIN` can be computed from probability columns.

Methods such as `SMP`, `PV`, `BALD`, and `ENT_MC` must already exist as columns in the prediction CSV if you are using `saved_scores` mode.

---

## 5. Evaluation outputs

Evaluation outputs are saved under the folder set in:

```text
outputs.outdir
```

For the English mBERT experiment, this is usually:

```text
results/english_validation_mbert/metrics/
```

Important files:

```text
fold_0/
  scores_raw.csv
  scores_for_metrics.csv
  metrics_wide.csv
  metrics_long.csv
  metric_times.csv

metrics_long_all_folds.csv
metrics_summary_by_method.csv
metrics_summary_wide_mean.csv
metrics_summary_wide_std.csv
metrics_summary_latex_ready.csv
all_folds_summary.csv
```

The main file to inspect is:

```text
metrics_summary_by_method.csv
```

Quick check:

```bash
python - <<'PY'
import pandas as pd

p = "results/english_validation_mbert/metrics/metrics_summary_by_method.csv"
df = pd.read_csv(p)
print(df.pivot(index="metric", columns="method", values="mean").round(4))
PY
```

---

## 6. Metric conventions

### Discrimination metrics

| Metric | Better |
|---|---|
| `ROC-AUC` | higher |
| `AU-PRC` | higher |

### Calibration metrics

| Metric | Better |
|---|---|
| `C-Slope` | closer to 1 |
| `CITL` | closer to 0 |
| `ECE` | lower |

### Selective prediction metrics

| Metric | Better |
|---|---|
| `RC-AUC` | higher |
| `Norm RC-AUC` | higher |
| `E-AUoptRC` | lower |
| `TI` | higher |
| `TI@95` | higher |
| `Optimal Coverage` | descriptive |

The repo uses the paper-style selective-prediction convention:

```text
RC-AUC = 1 - AURC
```

So `RC-AUC` is on a retained-performance scale, not a risk scale.

---

## 7. Reporting

After evaluation, run:

```bash
python scripts/make_report_outputs.py \
  --config configs/reporting_english_validation_mbert.yaml
```

### Where to change reporting file names

Edit this config file:

```text
configs/reporting_english_validation_mbert.yaml
```

Change these fields when moving to a new experiment:

| What to change | Config location | Example |
|---|---|---|
| experiment name | `experiment_name` | `english_validation_mbert` |
| language label | `lang` | `EN` |
| result folder | `paths.results_dir` | `results/english_validation_mbert` |
| metric summary input file | `paths.metrics_summary` | `results/.../metrics_summary_by_method.csv` |
| reporting output folder | `paths.output_dir` | `results/.../reporting` |
| input column names | `input.column_map` | map `method`, `metric`, `mean`, `std` |
| methods to include | `filters.methods` | `[SR, SMP, ENT]` |
| metrics to include | `filters.metrics` | selected metric names |
| table filename | `tables.filename` | `metrics_summary_en.tex` |
| table label | `tables.label` | `tab:metrics_summary_en` |
| figure formats | `plots.formats` | `[png, pdf]` |
| figure resolution | `plots.dpi` | `300` |
| lower-is-better metrics | `metric_rules.lower_is_better` | `[ECE, E-AUoptRC]` |
| target-based metrics | `metric_rules.target_is_best` | `CITL: 0.0`, `C-Slope: 1.0` |

### Reporting outputs

Reporting outputs are saved under:

```text
paths.output_dir
```

Usually:

```text
results/english_validation_mbert/reporting/
```

Typical output folders:

```text
csv/
tables/
figures/
```

---

## 8. Clean rerun

If you change metric definitions or config options, remove old summaries first:

```bash
rm -f results/english_validation_mbert/metrics/*summary*.csv
rm -f results/english_validation_mbert/metrics/*long*.csv
rm -f results/english_validation_mbert/reporting/csv/*.csv
rm -f results/english_validation_mbert/reporting/tables/*.tex
```

Then rerun evaluation and reporting:

```bash
python scripts/evaluate_saved_model_uncertainty.py \
  --config configs/evaluate_saved_model_uncertainty.yaml

python scripts/make_report_outputs.py \
  --config configs/reporting_english_validation_mbert.yaml
```

---

## 9. Full model-based scoring

Use this workflow only when you want the repo to load saved models and compute uncertainty scores.

Check files:

```bash
python scripts/run_folds.py \
  --config configs/english_mbert.yaml \
  --check-files
```

Run all configured folds:

```bash
python scripts/run_folds.py \
  --config configs/english_mbert.yaml
```

Run one fold:

```bash
python scripts/run_single_fold.py \
  --config configs/english_mbert.yaml \
  --fold 0
```

---

## 10. Adding a new uncertainty method

1. Add the method in:

```text
src/uncertainty_benchmark/methods/
```

2. Register it in:

```text
src/uncertainty_benchmark/methods/registry.py
```

3. Add it to the config:

```text
methods.enabled
```

4. Add or update tests.

---

## 11. Development checks

Run tests:

```bash
pytest tests
```

Check imports:

```bash
python - <<'PY'
from uncertainty_benchmark.metrics.suite import compute_metrics_per_method_with_timing
from uncertainty_benchmark.reporting import build_metric_summary_table
print("imports OK")
PY
```

---

## 12. Reproducibility

Each fold should use a fixed seed, normally:

```text
42 + fold_id
```

Timing code should use CUDA synchronization when a GPU is available.

---

## 13. Citation

If you use this repository, please cite:

```bibtex
@inproceedings{khallaf-sharoff-2026-predict,
  title     = {To Predict or Not to Predict? Towards Reliable Uncertainty Estimation in the Presence of Noise},
  author    = {Khallaf, Nouran and Sharoff, Serge},
  booktitle = {Proceedings of the International Conference on Language Resources and Evaluation (LREC)},
  year      = {2026}
}
```

The arXiv version is available at:

```text
https://arxiv.org/abs/2603.07330
```

---
