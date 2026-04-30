# Adding a New Uncertainty Method

## 1. Create the method class

Add your method in:

```text
src/uncertainty_benchmark/methods/
```

Example:

```python
import numpy as np

from uncertainty_benchmark.methods.base import UncertaintyMethod


class MarginUncertainty(UncertaintyMethod):
    name = "MARGIN"
    requires = ["eval_probs"]
    higher_is_uncertain = True

    def score(self, context):
        self.check_requirements(context)

        probs = np.asarray(context["eval_probs"], dtype=float)
        sorted_probs = np.sort(probs, axis=1)

        top1 = sorted_probs[:, -1]
        top2 = sorted_probs[:, -2]

        margin = top1 - top2
        uncertainty = 1.0 - margin

        return uncertainty
```

## 2. Register the method

Edit:

```text
src/uncertainty_benchmark/methods/registry.py
```

Add the class to `METHOD_REGISTRY`.

## 3. Enable it in config

```yaml
methods:
  enabled:
    - MARGIN
```

## 4. Add tests

Add a test in:

```text
tests/test_uncertainty_methods.py
```
