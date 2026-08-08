import io
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import app as app_module


@pytest.fixture
def client(tmp_path, monkeypatch):
    uploads_dir = tmp_path / "uploads"
    results_dir = tmp_path / "results"
    uploads_dir.mkdir(exist_ok=True)
    results_dir.mkdir(exist_ok=True)

    monkeypatch.setattr(app_module, "UPLOAD_FOLDER", str(uploads_dir))
    monkeypatch.setattr(app_module, "RESULT_FOLDER", str(results_dir))
    app_module.app.config["UPLOAD_FOLDER"] = str(uploads_dir)
    app_module.app.config["TESTING"] = True

    def fake_run_assessment(image_path, apply_enhancement=True):
        images = {
            "original": np.zeros((20, 20, 3), dtype=np.uint8),
            "enhanced": np.zeros((20, 20, 3), dtype=np.uint8),
            "mask": np.zeros((20, 20), dtype=np.uint8),
            "overlay": np.zeros((20, 20, 3), dtype=np.uint8),
            "gradcam": np.zeros((20, 20, 3), dtype=np.uint8),
        }
        result = {
            "damage_class": "Dent",
            "confidence": 0.91,
            "severity": "Minor",
            "severity_score": 0.25,
            "damage_percentage": 12.5,
            "shape_irregularity": 0.1,
            "recommendations": ["Paintless Dent Repair (PDR)"],
            "all_probs": {"Dent": 0.91, "Scratch": 0.09},
        }
        return images, result

    monkeypatch.setattr(app_module.inference, "run_assessment", fake_run_assessment)

    with app_module.app.test_client() as client:
        yield client


def test_index_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"AI Vehicle Damage Assessment" in response.data


def test_predict_route_handles_invalid_upload(client):
    response = client.post(
        "/predict",
        data={"image": (io.BytesIO(b"not-an-image"), "invalid.jpg")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert response.status_code == 200
    assert b"The uploaded file is not a valid image or is corrupt." in response.data


def test_predict_route_returns_results(client):
    image = np.zeros((30, 30, 3), dtype=np.uint8)
    _, encoded = cv2.imencode(".png", image)
    response = client.post(
        "/predict",
        data={"image": (io.BytesIO(encoded.tobytes()), "car.png"), "apply_enhancement": "on"},
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    assert b"Assessment Result" in response.data
    assert b"Dent" in response.data


def test_report_route_returns_pdf(client):
    results_dir = app_module.RESULT_FOLDER
    metadata_path = os.path.join(results_dir, "demo123.json")
    dummy_image_path = os.path.join(results_dir, "demo123_original.png")
    cv2.imwrite(dummy_image_path, np.zeros((10, 10, 3), dtype=np.uint8))

    metadata = {
        "run_id": "demo123",
        "uploaded_filename": "car.png",
        "result": {
            "damage_class": "Dent",
            "confidence": 0.91,
            "severity": "Minor",
            "damage_percentage": 12.5,
            "recommendations": ["Paintless Dent Repair (PDR)"],
        },
        "images": {
            "original": {"path": dummy_image_path},
            "enhanced": {"path": dummy_image_path},
            "overlay": {"path": dummy_image_path},
        },
    }
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh)

    response = client.get("/report/demo123")
    assert response.status_code == 200
    assert response.mimetype == "application/pdf"
    assert "damage_report_demo123.pdf" in response.headers["Content-Disposition"]
