# Uncertainty Methods

All methods return one score per evaluation item.

The convention is:

```text
larger score = more uncertain
```

## Deterministic methods

### SR

Softmax Response:

```text
SR = 1 - max_c p(y=c | x)
```

### ENT

Predictive entropy of the deterministic softmax distribution.

## MC-dropout methods

### SMP

Sampled Max Probability:

```text
SMP = 1 - max_c mean_t p_t(y=c | x)
```

### PV

Probability variance across MC-dropout samples.

### BALD

```text
BALD = H(E[p]) - E[H(p)]
```

### ENT_MC

Entropy of the mean MC-dropout predictive distribution.

## Distance methods

### MD

Minimum Mahalanobis distance from the evaluation logits to class centroids estimated from training logits.

## Hybrid methods

### HUQ-MD

Rank-based hybrid of:

- Mahalanobis Distance
- Softmax Response

## Outlier methods

### LOF

Local Outlier Factor over CLS embeddings.

### ISOF

Isolation Forest over CLS embeddings.
