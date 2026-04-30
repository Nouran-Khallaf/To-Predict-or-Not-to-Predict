from uncertainty_benchmark.config import load_config


def test_load_example_config():
    config = load_config("configs/example_config.yaml")
    assert "experiment_name" in config
    assert "methods" in config
