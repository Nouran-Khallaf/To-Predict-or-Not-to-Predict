# Reproducibility

## Seeding

Each fold uses:

```text
seed = 42 + fold_id
```

## Timing

Timing uses GPU synchronisation before reading the clock when CUDA is available.

## Score direction

Every uncertainty method returns scores where:

```text
larger = more uncertain
```

This avoids method-specific score flipping during metric computation.
