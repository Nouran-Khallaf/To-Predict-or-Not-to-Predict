install:
    pip install -e .

install-dev:
    pip install -e ".[dev]"

test:
    pytest tests

lint:
    ruff check src tests

format:
    black src tests scripts

run-english:
    python scripts/run_folds.py --config configs/english_mbert.yaml
