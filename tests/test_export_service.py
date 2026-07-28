import json

import pandas as pd
import pytest

from app.services import export_service


def test_validate_dataframe_none():
    with pytest.raises(export_service.EmptyDataError):
        export_service.validate_dataframe(None)


def test_validate_dataframe_empty():
    with pytest.raises(export_service.EmptyDataError):
        export_service.validate_dataframe(pd.DataFrame())


def test_validate_dataframe_valid(sample_dataset):
    export_service.validate_dataframe(sample_dataset)


def test_validate_export_format_csv():
    export_service.validate_export_format("csv")


def test_validate_export_format_json():
    export_service.validate_export_format("json")


def test_validate_export_format_excel():
    export_service.validate_export_format("xlsx")


def test_validate_export_format_invalid():
    with pytest.raises(export_service.UnsupportedFormatError):
        export_service.validate_export_format("pdf")


def test_sanitize_filename_spaces():
    result = export_service.sanitize_filename("my file name")
    assert result == "my_file_name"


def test_sanitize_filename_special_characters():
    result = export_service.sanitize_filename("my:file?.csv")
    assert ":" not in result
    assert "?" not in result


def test_generate_filename_without_timestamp():
    result = export_service.generate_filename(
        "predictions",
        "csv",
        include_timestamp=False,
    )

    assert result == "predictions.csv"


def test_generate_filename_with_timestamp():
    result = export_service.generate_filename(
        "predictions",
        "csv",
    )

    assert result.startswith("predictions_")
    assert result.endswith(".csv")


def test_dataframe_to_csv(sample_dataset):

    csv_bytes = export_service.dataframe_to_csv(sample_dataset)

    assert isinstance(csv_bytes, bytes)

    text = csv_bytes.decode()

    assert "FeatureA" in text
    assert "FeatureB" in text
    assert "Churn" in text


def test_dataframe_to_json(sample_dataset):

    json_bytes = export_service.dataframe_to_json(sample_dataset)

    assert isinstance(json_bytes, bytes)

    data = json.loads(json_bytes.decode())

    assert isinstance(data, list)
    assert len(data) == len(sample_dataset)

def test_dataframe_to_excel(sample_dataset):

    excel_bytes = export_service.dataframe_to_excel(sample_dataset)

    assert isinstance(excel_bytes, bytes)
    assert len(excel_bytes) > 0


def test_dictionary_to_json():

    payload = {
        "accuracy": 0.95,
        "precision": 0.92,
    }

    result = export_service.dictionary_to_json(payload)

    decoded = json.loads(result.decode())

    assert decoded == payload


def test_export_dataframe_csv(sample_dataset):

    data, filename, mime = export_service.export_dataframe(
        sample_dataset,
        "dataset",
        "csv",
    )

    assert isinstance(data, bytes)
    assert filename.endswith(".csv")
    assert mime == export_service.CSV_MIME


def test_export_dataframe_json(sample_dataset):

    data, filename, mime = export_service.export_dataframe(
        sample_dataset,
        "dataset",
        "json",
    )

    assert isinstance(data, bytes)
    assert filename.endswith(".json")
    assert mime == export_service.JSON_MIME


def test_export_dataframe_excel(sample_dataset):

    data, filename, mime = export_service.export_dataframe(
        sample_dataset,
        "dataset",
        "xlsx",
    )

    assert isinstance(data, bytes)
    assert filename.endswith(".xlsx")
    assert mime == export_service.EXCEL_MIME


def test_export_dataframe_invalid_format(sample_dataset):

    with pytest.raises(export_service.UnsupportedFormatError):
        export_service.export_dataframe(
            sample_dataset,
            "dataset",
            "xml",
        )

def test_export_predictions_csv(sample_dataset):

    data, filename, mime = export_service.export_predictions_csv(
        sample_dataset
    )

    assert isinstance(data, bytes)
    assert filename.endswith(".csv")
    assert mime == export_service.CSV_MIME


def test_export_predictions_excel(sample_dataset):

    data, filename, mime = export_service.export_predictions_excel(
        sample_dataset
    )

    assert isinstance(data, bytes)
    assert filename.endswith(".xlsx")
    assert mime == export_service.EXCEL_MIME


def test_export_predictions_json(sample_dataset):

    data, filename, mime = export_service.export_predictions_json(
        sample_dataset
    )

    assert isinstance(data, bytes)
    assert filename.endswith(".json")
    assert mime == export_service.JSON_MIME


def test_export_evaluation_metrics():

    metrics = {
        "accuracy": 0.95,
        "precision": 0.93,
    }

    data, filename, mime = export_service.export_evaluation_metrics(
        metrics
    )

    assert isinstance(data, bytes)
    assert filename.endswith(".json")
    assert mime == export_service.JSON_MIME


def test_export_model_comparison(sample_dataset):

    data, filename, mime = export_service.export_model_comparison(
        sample_dataset
    )

    assert isinstance(data, bytes)
    assert filename.endswith(".csv")
    assert mime == export_service.CSV_MIME


def test_export_classification_report(sample_dataset):

    data, filename, mime = export_service.export_classification_report(
        sample_dataset
    )

    assert isinstance(data, bytes)
    assert filename.endswith(".csv")
    assert mime == export_service.CSV_MIME


def test_export_feature_importance(sample_dataset):

    data, filename, mime = export_service.export_feature_importance(
        sample_dataset
    )

    assert isinstance(data, bytes)
    assert filename.endswith(".xlsx")
    assert mime == export_service.EXCEL_MIME


def test_export_shap_values(sample_dataset):

    data, filename, mime = export_service.export_shap_values(
        sample_dataset
    )

    assert isinstance(data, bytes)
    assert filename.endswith(".xlsx")
    assert mime == export_service.EXCEL_MIME

def test_build_export_package():

    package = export_service.build_export_package(
        b"hello",
        "demo.csv",
        export_service.CSV_MIME,
    )

    assert package["data"] == b"hello"
    assert package["filename"] == "demo.csv"
    assert package["mime_type"] == export_service.CSV_MIME
    assert package["size"] == 5


def test_prepare_dataframe_export(sample_dataset):

    package = export_service.prepare_dataframe_export(
        sample_dataset,
        "dataset",
        "csv",
    )

    assert package["filename"].endswith(".csv")
    assert package["mime_type"] == export_service.CSV_MIME
    assert package["size"] > 0


def test_prepare_prediction_export(sample_dataset):

    package = export_service.prepare_prediction_export(
        sample_dataset
    )

    assert package["filename"].endswith(".csv")
    assert package["size"] > 0


def test_prepare_model_comparison_export(sample_dataset):

    package = export_service.prepare_model_comparison_export(
        sample_dataset
    )

    assert package["filename"].endswith(".csv")


def test_prepare_metrics_export():

    package = export_service.prepare_metrics_export(
        {"accuracy": 1.0}
    )

    assert package["filename"].endswith(".json")


def test_get_supported_export_formats():

    formats = export_service.get_supported_export_formats()

    assert formats == sorted(
        export_service.SUPPORTED_EXPORT_FORMATS
    )


def test_get_export_statistics():

    package = {
        "filename": "demo.csv",
        "mime_type": export_service.CSV_MIME,
        "size": 2048,
    }

    stats = export_service.get_export_statistics(package)

    assert stats["filename"] == "demo.csv"
    assert stats["mime_type"] == export_service.CSV_MIME
    assert stats["size_bytes"] == 2048
    assert stats["size_kb"] == 2.0


def test_get_export_service_information():

    info = export_service.get_export_service_information()

    assert info["service"] == "Export Service"
    assert "version" in info
    assert "supported_formats" in info
    assert "default_date_format" in info

def test_log_export_success(monkeypatch):

    called = {}

    def fake_info(*args, **kwargs):
        called["info"] = True

    monkeypatch.setattr(
        export_service.logger,
        "info",
        fake_info,
    )

    export_service.log_export_success(
        "demo.csv",
        100,
    )

    assert called["info"]


def test_log_export_failure(monkeypatch):

    called = {}

    def fake_exception(*args, **kwargs):
        called["exception"] = True

    monkeypatch.setattr(
        export_service.logger,
        "exception",
        fake_exception,
    )

    export_service.log_export_failure(
        "csv",
        Exception("failure"),
    )

    assert called["exception"]