"""Integration tests for the waste classification API."""

import math
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app

EXPECTED_CLASSES = {
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash",
}

UPLOAD_LIMIT = 5 * 1024 * 1024


@pytest.fixture(scope="module")
def client():
    """Start the application once for this test module."""
    with TestClient(app) as test_client:
        yield test_client


def make_image_bytes(image_format):
    """Create a small image in memory without using personal photos."""
    with BytesIO() as buffer:
        with Image.new("RGB", (160, 120), color=(180, 140, 90)) as image:
            image.save(buffer, format=image_format)
        return buffer.getvalue()


def test_homepage_returns_html(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "<html" in response.text.lower()


def test_health_reports_model_ready(client):
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_ready"] is True


@pytest.mark.parametrize(
    "image_format, filename, content_type",
    [
        ("JPEG", "sample.jpg", "image/jpeg"),
        ("PNG", "sample.png", "image/png"),
    ],
)
def test_valid_image_returns_prediction(client, image_format, filename, content_type):
    image_bytes = make_image_bytes(image_format)

    response = client.post(
        "/predict",
        files={"file": (filename, image_bytes, content_type)},
    )

    assert response.status_code == 200, response.text

    body = response.json()
    scores = body["class_scores"]

    assert body["model"] == "mobilenet_v2_finetuned"
    assert set(scores) == EXPECTED_CLASSES
    assert body["predicted_class"] in EXPECTED_CLASSES

    for score in scores.values():
        assert isinstance(score, (int, float))
        assert math.isfinite(score)
        assert 0.0 <= score <= 1.0

    assert sum(scores.values()) == pytest.approx(1.0, abs=1e-5)

    confidence = body["confidence"]
    assert confidence == pytest.approx(scores[body["predicted_class"]])
    assert confidence == pytest.approx(max(scores.values()))


def test_text_disguised_as_jpeg_is_rejected(client):
    response = client.post(
        "/predict",
        files={
            "file": (
                "fake.jpg",
                b"This is text, not an image.",
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == ("The file is not a valid readable image.")


def test_empty_upload_is_rejected(client):
    response = client.post(
        "/predict",
        files={"file": ("empty.jpg", b"", "image/jpeg")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "The uploaded file is empty."


def test_oversized_upload_is_rejected(client):
    oversized_bytes = b"x" * (UPLOAD_LIMIT + 1)

    response = client.post(
        "/predict",
        files={
            "file": (
                "large.jpg",
                oversized_bytes,
                "image/jpeg",
            )
        },
    )

    assert response.status_code == 413
    assert response.json()["detail"] == ("Maximum supported file size is 5 MiB.")


def test_missing_file_is_rejected(client):
    response = client.post("/predict")

    assert response.status_code == 422
    assert any(error["loc"] == ["body", "file"] for error in response.json()["detail"])


def test_unsupported_image_format_is_rejected(client):
    image_bytes = make_image_bytes("BMP")

    response = client.post(
        "/predict",
        files={"file": ("sample.bmp", image_bytes, "image/bmp")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == ("Only JPEG and PNG images are supported.")


def test_prediction_without_model_returns_503(client, monkeypatch):
    # Temporarily simulate an unavailable predictor.
    # monkeypatch restores the original value after this test.
    monkeypatch.setattr(app.state, "predictor", None)

    response = client.post(
        "/predict",
        files={
            "file": (
                "sample.png",
                make_image_bytes("PNG"),
                "image/png",
            )
        },
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Model is not ready."
