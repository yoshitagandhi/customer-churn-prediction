from pathlib import Path

import pandas as pd

from app.utils import cache


def test_get_cached_model(monkeypatch):
    expected = object()

    def fake_load_model(path):
        assert isinstance(path, Path)
        return expected

    monkeypatch.setattr(cache, "load_model", fake_load_model)

    result = cache.get_cached_model()

    assert result is expected


def test_get_cached_training_metadata(monkeypatch, sample_metadata):
    monkeypatch.setattr(
        cache,
        "load_training_metadata",
        lambda path: sample_metadata,
    )

    result = cache.get_cached_training_metadata()

    assert result == sample_metadata

def test_get_cached_threshold_config(monkeypatch):
    config = {"optimal_threshold": 0.5}

    monkeypatch.setattr(
        cache,
        "load_threshold_config",
        lambda path: config,
    )

    result = cache.get_cached_threshold_config()

    assert result == config

def test_get_cached_dataset(monkeypatch, sample_dataset):
    monkeypatch.setattr(
        cache,
        "load_dataset",
        lambda: sample_dataset,
    )

    monkeypatch.setattr(
        cache,
        "clean_dataset",
        lambda df: df,
    )

    result = cache.get_cached_dataset()

    assert result.equals(sample_dataset)


def test_get_cached_validation_dataset(monkeypatch, sample_dataset):
    monkeypatch.setattr(
        cache,
        "get_cached_dataset",
        lambda: sample_dataset,
    )

    features, target = cache.get_cached_validation_dataset()

    assert "Churn" not in features.columns
    assert len(features) == len(target)
    assert target.tolist() == sample_dataset["Churn"].tolist()


def test_background_sample_small(monkeypatch, sample_dataset):
    monkeypatch.setattr(
        cache,
        "get_cached_dataset",
        lambda: sample_dataset,
    )

    result = cache.get_cached_background_sample(sample_size=10)

    assert len(result) == len(sample_dataset)
    assert "Churn" not in result.columns

def test_background_sample_large(monkeypatch, large_dataset):
    monkeypatch.setattr(
        cache,
        "get_cached_dataset",
        lambda: large_dataset,
    )

    result = cache.get_cached_background_sample(sample_size=100)

    assert len(result) == 100
    assert "Churn" not in result.columns

def test_get_cached_explainer(monkeypatch):
    pipeline = object()

    monkeypatch.setattr(
        cache,
        "get_cached_background_sample",
        lambda: "background",
    )

    monkeypatch.setattr(
        cache,
        "get_processed_features",
        lambda p, b: "processed",
    )

    monkeypatch.setattr(
        cache,
        "load_explainer",
        lambda p, f: "explainer",
    )

    result = cache.get_cached_explainer(pipeline)

    assert result == "explainer"

def test_get_cached_experiment_records(monkeypatch, sample_metadata):
    monkeypatch.setattr(
        cache,
        "get_cached_training_metadata",
        lambda: sample_metadata,
    )

    result = cache.get_cached_experiment_records()

    assert result == sample_metadata["experiment_records"]


def test_get_cached_feature_metadata(monkeypatch, sample_metadata):
    monkeypatch.setattr(
        cache,
        "get_cached_training_metadata",
        lambda: sample_metadata,
    )

    result = cache.get_cached_feature_metadata()

    assert result == sample_metadata["feature_metadata"]


def test_clear_all_cache(monkeypatch):
    calls = {
        "data": False,
        "resource": False,
    }

    def fake_clear_data():
        calls["data"] = True

    def fake_clear_resource():
        calls["resource"] = True

    monkeypatch.setattr(cache.st.cache_data, "clear", fake_clear_data)
    monkeypatch.setattr(cache.st.cache_resource, "clear", fake_clear_resource)

    cache.clear_all_cache()

    assert calls["data"] is True
    assert calls["resource"] is True