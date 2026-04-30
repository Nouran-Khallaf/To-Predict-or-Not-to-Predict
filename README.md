# Uncertainty Benchmark

A small benchmark for evaluating uncertainty estimation methods for text classification and readability prediction.

The repository supports three main workflows:

1. **Evaluate saved prediction CSVs**  
   Use this when you already have fold-level prediction files with labels, probabilities, and optionally saved uncertainty scores such as `SMP`.

2. **Fit rejection thresholds / selective prediction curves**  
   Use this after uncertainty scores have been computed. This finds thresholds for accepting or rejecting predictions based on uncertainty.

3. **Generate reporting outputs**  
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
│   ├── rejection_fit.yaml
│   ├── smp_rejection_apply.yaml
│   └── reporting_english_validation_mbert.yaml
├── data/
├── results/
├── scripts/
│   ├── evaluate_saved_model_uncertainty.py
│   ├── smp_rejection_thresholds.py
│   ├── summarise_rejection_curve.py
│   ├── summarise_rejection_by_coverage.py
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

If imports fail, add `src` to `PYTHONPATH`:

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

Methods such as `SMP`, `PV`, `BALD`, and `ENT_MC` can be evaluated in two ways. In saved-score mode, they must already exist as columns in the prediction CSV. In model-scoring mode, the script loads the saved fold model and computes MC-dropout uncertainty scores before evaluation.

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

---

## 7. Rejection thresholds and selective prediction

After uncertainty scores have been computed, the repository can evaluate selective prediction.

Selective prediction asks:

```text
Can the model become more reliable if it refuses to predict on the most uncertain examples?
```

The idea is simple:

```text
larger uncertainty score = less reliable prediction
```

For a method such as `SMP`, `SR`, or `ENT`, we choose a threshold:

```text
accept prediction if uncertainty <= threshold
reject prediction if uncertainty > threshold
```

The accepted examples are the examples the model keeps. The rejected examples are the examples that would be sent to manual review, a fallback model, or a safer downstream process.

### 7.1 Fit rejection thresholds

Use this after running uncertainty evaluation and saving score files.

For the English validation mBERT experiment, the input score files are:

```text
results/english_validation_mbert/scores/fold_*_scores_wide.csv
```

Each file should contain at least:

```text
true_label
predicted_label
SMP
SR
ENT
```

Check this with:

```bash
head -1 results/english_validation_mbert/scores/fold_0_scores_wide.csv
```

The threshold config is:

```text
configs/rejection_fit.yaml
```

Example config:

```yaml
mode: fit_thresholds

task_type: classification

input_predictions_glob: "results/english_validation_mbert/scores/fold_*_scores_wide.csv"

output_thresholds: "results/english_validation_mbert/rejection_thresholds.csv"
output_curve: "results/english_validation_mbert/rejection_curve.csv"

columns:
  gold_col: "true_label"
  pred_col: "predicted_label"
  language_col: null

uncertainty:
  methods: ["SMP", "SR", "ENT"]
  score_direction: "higher_is_uncertain"

languages:
  use: "all"

threshold:
  scope: "global"

threshold_selection:
  mode: "target_coverage"
  target_coverage: 0.80

coverage_grid:
  start: 1.00
  end: 0.50
  step: 0.05

bootstrap:
  enabled: true
  n_resamples: 200
  ci: 0.95
  random_seed: 42
```

Run:

```bash
python scripts/smp_rejection_thresholds.py \
  --config configs/rejection_fit.yaml
```

This produces:

```text
results/english_validation_mbert/rejection_thresholds.csv
results/english_validation_mbert/rejection_curve.csv
```

### 7.2 What the threshold means

For a target coverage of 80%, the script chooses a threshold that keeps about 80% of examples and rejects about 20%.

For example, if the selected SMP threshold is:

```text
0.272666
```

then the decision rule is:

```text
accept if SMP <= 0.272666
reject if SMP > 0.272666
```

This is because the score convention is:

```text
higher score = more uncertain
```

The threshold does not change the model itself. It only changes which predictions are considered safe enough to keep.

### 7.3 Summarise one coverage point

To summarise the results at 80% coverage, run:

```bash
python scripts/summarise_rejection_curve.py \
  --curve results/english_validation_mbert/rejection_curve.csv \
  --target_coverage 0.80 \
  --output results/english_validation_mbert/rejection_summary_at_80.csv
```

This prints a table like:

```text
At around 80% coverage, the results are:

Method  Accepted  Rejected  Threshold  Accepted accuracy
SMP        1395       349   0.272666             0.9075
SR         1395       349   0.285991             0.9039
ENT        1395       349   0.598523             0.9039
```

Interpretation:

```text
Without rejection: accuracy = 0.8825
With SMP rejection at 80% coverage: accepted accuracy = 0.9075
Rejected examples = 349 / 1744 ≈ 20%
```

This means the model is more accurate on the subset of predictions it keeps.

### 7.4 Summarise several coverage levels

To compare the best method at several coverage levels, run:

```bash
python scripts/summarise_rejection_by_coverage.py \
  --curve results/english_validation_mbert/rejection_curve.csv \
  --coverages 0.90 0.80 0.70 0.60 0.50 \
  --output results/english_validation_mbert/rejection_best_by_coverage.csv
```

For the English validation mBERT experiment, the results were:

| Target coverage | Best method | Accepted accuracy | Gain over no rejection | Accepted | Rejected |
|---:|---|---:|---:|---:|---:|
| 90% | `ENT` | 0.9063 | +0.0239 | 1569 | 175 |
| 80% | `SMP` | 0.9075 | +0.0251 | 1395 | 349 |
| 70% | `ENT` | 0.9500 | +0.0675 | 1220 | 524 |
| 60% | `SMP` | 0.9522 | +0.0697 | 1046 | 698 |
| 50% | `ENT` | 0.9610 | +0.0786 | 872 | 872 |

The no-rejection baseline accuracy was:

```text
0.8825
```

This shows that rejection improves accepted-set accuracy at every tested coverage level.

### 7.5 How to describe the result

A short report sentence:

```text
Selective prediction improves reliability by trading coverage for accuracy. The full-coverage model achieves 0.8825 accuracy. At 80% coverage, SMP gives the strongest operating point, increasing accepted accuracy to 0.9075 while rejecting 20% of examples. Under stricter rejection, accepted accuracy rises further, reaching 0.9610 at 50% coverage with entropy, although only half of the examples are accepted.
```

A more careful version:

```text
The best uncertainty method depends on the rejection budget. Entropy performs best at 90%, 70%, and 50% coverage, while SMP performs best at 80% and 60% coverage. This suggests that MC-dropout uncertainty is useful at some operating points, but deterministic probability-based uncertainty remains competitive.
```

### 7.6 Optional: apply a fitted threshold

After fitting thresholds, apply the selected threshold to the score files:

```bash
python scripts/smp_rejection_thresholds.py \
  --config configs/smp_rejection_apply.yaml
```

Example apply config:

```yaml
mode: apply_thresholds

task_type: classification

input_predictions_glob: "results/english_validation_mbert/scores/fold_*_scores_wide.csv"

input_thresholds: "results/english_validation_mbert/rejection_thresholds.csv"

output_predictions: "results/english_validation_mbert/smp_rejection_applied.csv"
output_summary: "results/english_validation_mbert/smp_rejection_summary.csv"

columns:
  gold_col: "true_label"
  pred_col: "predicted_label"
  language_col: null

uncertainty:
  methods: ["SMP"]
  score_direction: "higher_is_uncertain"

threshold:
  scope: "global"
```

The applied output marks each prediction as accepted or rejected according to the fitted threshold.

---

## 8. Reporting

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

## 9. Full model-based scoring

Use this workflow when you want the repo to load saved models and compute uncertainty scores.

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

## 12. Citation

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
