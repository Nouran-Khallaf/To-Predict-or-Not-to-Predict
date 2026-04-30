# Configuration

Experiments are controlled by YAML files in:

```text
configs/
```

## Fold control

Run one fold:

```yaml
folds:
  n_folds: 1
  fold_ids: [0]
```

Run selected folds:

```yaml
folds:
  fold_ids: [0, 2, 4]
```

Run the first `n` folds:

```yaml
folds:
  n_folds: 5
  fold_ids: null
```

## Method control

Run deterministic methods only:

```yaml
methods:
  enabled:
    - SR
    - ENT
```

Run all methods:

```yaml
methods:
  enabled:
    - SR
    - ENT
    - SMP
    - PV
    - BALD
    - ENT_MC
    - MD
    - HUQ-MD
    - LOF
    - ISOF
```

## MC-dropout settings

```yaml
mc_dropout:
  committee_size: 20
  dropout_p: 0.10
```

## Embedding settings

```yaml
embeddings:
  batch_size: 32
```

## Metrics

```yaml
metrics:
  ece_bins: 15
  ti_fixed_coverage: 0.95
```
