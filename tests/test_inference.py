import numpy as np
import pytest

import inference


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def test_load_models_raises_if_missing(monkeypatch):
    monkeypatch.setattr(inference, "UNET_PATH", "/tmp/does-not-exist-unet.keras")
    monkeypatch.setattr(inference, "CLF_PATH", "/tmp/does-not-exist-clf.keras")

    def fake_hf_download(*args, **kwargs):
        raise FileNotFoundError("Model not found")

    monkeypatch.setattr(inference, "hf_hub_download", fake_hf_download)

    try:
        inference.load_models()
    except FileNotFoundError as exc:
        assert "Model not found" in str(exc)
    else:
        raise AssertionError("load_models() should raise FileNotFoundError")


# ---------------------------------------------------------------------------
# assess_damage_severity
# ---------------------------------------------------------------------------
def test_assess_damage_severity_minor():
    mask = np.zeros((32, 32), dtype=np.uint8)
    result = inference.assess_damage_severity(mask, confidence=0.5, damage_class="Dent")
    assert result["severity"] == "Minor"
    assert result["damage_percentage"] == 0.0


def test_assess_damage_severity_zero_mask():
    """64x64 all-zeros mask → Minor severity, 0% damaged area."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    result = inference.assess_damage_severity(mask, confidence=0.5, damage_class="Dent")
    assert result["severity"] == "Minor"
    assert result["damage_percentage"] == pytest.approx(0.0)


def test_assess_damage_severity_full_mask():
    """64x64 all-ones mask with high confidence → Severe severity."""
    mask = np.ones((64, 64), dtype=np.uint8)
    result = inference.assess_damage_severity(mask, confidence=0.95, damage_class="Dent")
    assert result["severity"] == "Severe"


def test_assess_damage_severity_glass_shatter_override():
    """Domain override: Glass Shatter with zero-damage mask must not be Minor."""
    mask = np.zeros((32, 32), dtype=np.uint8)
    result = inference.assess_damage_severity(mask, confidence=0.3, damage_class="Glass Shatter")
    assert result["severity"] == "Moderate"


def test_glass_shatter_domain_override():
    """64x64 zeros mask: Glass Shatter domain rule prevents 'Minor' rating."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    result = inference.assess_damage_severity(mask, confidence=0.3, damage_class="Glass Shatter")
    assert result["severity"] in ("Moderate", "Severe"), (
        f"Expected Moderate or Severe for Glass Shatter, got {result['severity']}"
    )


# ---------------------------------------------------------------------------
# get_repair_recommendation
# ---------------------------------------------------------------------------
def test_get_repair_recommendation_known_class():
    result = inference.get_repair_recommendation("Scratch", "Minor")
    assert isinstance(result, list)
    assert len(result) > 0


def test_get_repair_recommendation_known():
    """Spec alias: known class returns a non-empty list."""
    result = inference.get_repair_recommendation("Scratch", "Minor")
    assert isinstance(result, list)
    assert len(result) > 0


def test_get_repair_recommendation_unknown_class():
    result = inference.get_repair_recommendation("Explosion", "Severe")
    assert result == ["Manual inspection recommended"]


def test_get_repair_recommendation_unknown():
    """Spec alias: unknown class gracefully returns fallback."""
    result = inference.get_repair_recommendation("Explosion", "Severe")
    assert result == ["Manual inspection recommended"] or len(result) > 0


# ---------------------------------------------------------------------------
# enhance_image_adaptive
# ---------------------------------------------------------------------------
def test_enhance_image_adaptive_passthrough():
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    output = inference.enhance_image_adaptive(img, enable=False)
    assert np.array_equal(output, img)


# ---------------------------------------------------------------------------
# Health endpoint (Flask test client — no model loading required)
# ---------------------------------------------------------------------------
def test_health_endpoint():
    """GET /health returns 200 and JSON body {"status": "ok"}."""
    import app as app_module
    app_module.app.config["TESTING"] = True
    with app_module.app.test_client() as client:
        response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data == {"status": "ok"}
