import pandas as pd

from app.services import dataset_service

def test_get_dataset(monkeypatch):

    dataframe = pd.DataFrame(
        {
            "A": [1, 2],
            "B": [3, 4],
        }
    )

    monkeypatch.setattr(
        dataset_service,
        "load_dataset",
        lambda path: dataframe,
    )

    result = dataset_service.get_dataset(
        "dummy.csv"
    )

    assert result.equals(dataframe)


def test_get_dataset_profile(monkeypatch):

    dataframe = pd.DataFrame({"A": [1]})

    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda path: dataframe,
    )

    monkeypatch.setattr(
        dataset_service,
        "profile_dataset",
        lambda df: {"missing": 0},
    )

    result = dataset_service.get_dataset_profile(
        "dummy.csv"
    )

    assert result == {"missing": 0}


def test_get_dataset_summary(monkeypatch):

    dataframe = pd.DataFrame(
        {
            "A": [1, 2],
            "B": [3, 4],
        }
    )

    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda path: dataframe,
    )

    monkeypatch.setattr(
        dataset_service,
        "get_dataset_profile",
        lambda path: {"quality": "good"},
    )

    summary = dataset_service.get_dataset_summary(
        "dummy.csv"
    )

    assert summary["rows"] == 2
    assert summary["columns"] == 2
    assert summary["profile"] == {"quality": "good"}
    assert isinstance(summary["memory_mb"], float)

def test_get_dataset_preview(monkeypatch):

    dataframe = pd.DataFrame(
        {
            "A": range(20),
        }
    )

    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda path: dataframe,
    )

    preview = dataset_service.get_dataset_preview(
        "dummy.csv",
        rows=5,
    )

    assert len(preview) == 5
    assert preview.iloc[0]["A"] == 0


def test_get_column_names(monkeypatch):

    dataframe = pd.DataFrame(
        {
            "Age": [25],
            "Income": [50000],
        }
    )

    monkeypatch.setattr(
        dataset_service,
        "get_dataset",
        lambda path: dataframe,
    )

    columns = dataset_service.get_column_names(
        "dummy.csv"
    )

    assert columns == [
        "Age",
        "Income",
    ]