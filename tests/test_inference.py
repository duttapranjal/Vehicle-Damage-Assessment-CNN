import numpy as np

import inference


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


def test_assess_damage_severity_minor():
    mask = np.zeros((32, 32), dtype=np.uint8)
    result = inference.assess_damage_severity(mask, confidence=0.5, damage_class="Dent")
    assert result["severity"] == "Minor"
    assert result["damage_percentage"] == 0.0


def test_assess_damage_severity_glass_shatter_override():
    mask = np.zeros((32, 32), dtype=np.uint8)
    result = inference.assess_damage_severity(mask, confidence=0.3, damage_class="Glass Shatter")
    assert result["severity"] == "Moderate"


def test_get_repair_recommendation_known_class():
    result = inference.get_repair_recommendation("Scratch", "Minor")
    assert isinstance(result, list)
    assert len(result) > 0


def test_get_repair_recommendation_unknown_class():
    result = inference.get_repair_recommendation("Explosion", "Severe")
    assert result == ["Manual inspection recommended"]


def test_enhance_image_adaptive_passthrough():
    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    output = inference.enhance_image_adaptive(img, enable=False)
    assert np.array_equal(output, img)
