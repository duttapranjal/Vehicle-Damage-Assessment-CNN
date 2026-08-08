"""
inference.py
------------
This module is a direct port of the pipeline built in your notebook
(Vehicle_Damage_Assessment_BootCamp_Project_Final.ipynb):

    enhance_image_adaptive -> U-Net segmentation -> extract_roi ->
    EfficientNetB0 classification -> severity scoring -> Grad-CAM ->
    repair recommendation

It loads your two trained models once at startup and exposes a single
function, run_assessment(image_path), that the Flask app calls.
"""

import os

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from huggingface_hub import hf_hub_download

# ---------------------------------------------------------------------------
# Config -- must match the notebook
# ---------------------------------------------------------------------------
IMG_SIZE = 224
SEG_SIZE = 256
CLASS_NAMES = ["Dent", "Scratch", "Crack", "Glass Shatter", "Lamp Broken", "Tire Flat"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

UNET_PATH = os.path.join(MODEL_DIR, "unet_final.keras")
CLF_PATH = os.path.join(MODEL_DIR, "efficientnet_classifier_final.keras")
HF_REPO_ID = os.environ.get("HF_REPO_ID", "YOUR_HF_USERNAME/vehicle-damage-models")
HF_TOKEN = os.environ.get("HF_TOKEN", None)   # only needed if repo is private

# ---------------------------------------------------------------------------
# Load models once, at import time
# ---------------------------------------------------------------------------
_unet_model = None
_clf_model = None
_last_conv_layer_name = None


@tf.keras.utils.register_keras_serializable(package="Custom")
def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    smooth = 1.0
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)


@tf.keras.utils.register_keras_serializable(package="Custom")
def bce_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
    dice = dice_coefficient(y_true, y_pred)
    return bce + (1.0 - dice)


def _find_last_conv_layer_name(model, min_rank=4):
    for layer in reversed(model.layers):
        try:
            if len(layer.output.shape) == min_rank:
                return layer.name
        except AttributeError:
            continue
    raise ValueError("Could not find a conv-like layer for Grad-CAM.")


def load_models():
    """Loads both .keras models into memory. Called once when the Flask app starts."""
    global _unet_model, _clf_model, _last_conv_layer_name
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR, exist_ok=True)

    def _resolve_model(local_path: str, hf_filename: str) -> str:
        if os.path.exists(local_path):
            print(f"[inference] Using local model: {local_path}")
            return local_path
        print(f"[inference] Downloading {hf_filename} from HuggingFace Hub ...")
        return hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=hf_filename,
            token=HF_TOKEN,
            cache_dir=os.path.join(BASE_DIR, ".hf_cache"),
        )

    unet_path = _resolve_model(UNET_PATH, "unet_final.keras")
    clf_path = _resolve_model(CLF_PATH, "efficientnet_classifier_final.keras")

    _unet_model = tf.keras.models.load_model(
        unet_path,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_coefficient": dice_coefficient,
        },
        compile=False,
    )
    _clf_model = tf.keras.models.load_model(clf_path, compile=False)
    _last_conv_layer_name = _find_last_conv_layer_name(_clf_model)
    print(f"[inference] Models ready. Grad-CAM layer: {_last_conv_layer_name}")


# ---------------------------------------------------------------------------
# Classical preprocessing / adaptive enhancement (Part 4 + Feature additions)
# ---------------------------------------------------------------------------
def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge((l_eq, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def apply_bilateral_filter(img_rgb):
    return cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)


def apply_contrast_enhancement(img_rgb, alpha=1.3, beta=10):
    return cv2.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)


def gamma_correction(img_rgb, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(img_rgb, table)


def brightness_normalization(img_rgb, target_mean=130):
    current_mean = img_rgb.mean()
    if current_mean < 1e-6:
        return img_rgb
    shift = target_mean - current_mean
    return np.clip(img_rgb.astype(np.float32) + shift, 0, 255).astype(np.uint8)


def auto_gamma_from_brightness(img_rgb):
    mean_intensity = img_rgb.mean()
    if mean_intensity < 80:
        return 1.6
    elif mean_intensity < 120:
        return 1.25
    elif mean_intensity > 190:
        return 0.8
    return 1.0


def enhance_image_adaptive(
    img_rgb,
    enable=True,
    use_clahe=True,
    use_gamma=True,
    use_brightness_norm=True,
    use_contrast=True,
    use_denoise=True,
):
    if not enable:
        return img_rgb.copy()

    out = img_rgb.copy()
    if use_brightness_norm:
        out = brightness_normalization(out)
    if use_gamma:
        out = gamma_correction(out, gamma=auto_gamma_from_brightness(out))
    if use_clahe:
        out = apply_clahe(out)
    if use_contrast:
        out = apply_contrast_enhancement(out)
    if use_denoise:
        out = apply_bilateral_filter(out)
    return out


# ---------------------------------------------------------------------------
# ROI extraction (Part 6, unchanged)
# ---------------------------------------------------------------------------
def extract_roi(img_rgb, binary_mask, pad_ratio=0.1):
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return img_rgb

    x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    h, w = binary_mask.shape
    pad_x, pad_y = int((x2 - x1) * pad_ratio), int((y2 - y1) * pad_ratio)
    x1, x2 = max(0, x1 - pad_x), min(w, x2 + pad_x)
    y1, y2 = max(0, y1 - pad_y), min(h, y2 + pad_y)

    if x2 <= x1 or y2 <= y1:
        return img_rgb

    return img_rgb[y1:y2, x1:x2]


# ---------------------------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------------------------
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    final_pred_index = pred_index.numpy() if hasattr(pred_index, "numpy") else pred_index
    return heatmap.numpy(), int(final_pred_index)


def overlay_heatmap(img_rgb_uint8, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img_rgb_uint8.shape[1], img_rgb_uint8.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_rgb_uint8, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


# ---------------------------------------------------------------------------
# Severity scoring (rule-based)
# ---------------------------------------------------------------------------
SEVERITY_THRESHOLDS = {"minor_max": 0.33, "moderate_max": 0.66}
SEVERITY_WEIGHTS = {"area": 0.60, "shape": 0.25, "confidence": 0.15}


def compute_shape_irregularity(binary_mask):
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < 1:
        return 0.0
    hull = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    if hull_area < 1:
        return 0.0
    solidity = area / hull_area
    return float(np.clip(1 - solidity, 0, 1))


def assess_damage_severity(binary_mask, confidence, damage_class=None):
    total_px = binary_mask.size
    damage_px = int(binary_mask.sum())
    damage_percentage = (damage_px / total_px) * 100 if total_px > 0 else 0.0

    area_score = float(np.clip(damage_percentage / 25.0, 0, 1))
    shape_score = compute_shape_irregularity(binary_mask)
    confidence_score = float(np.clip(confidence, 0, 1))

    severity_score = (
        SEVERITY_WEIGHTS["area"] * area_score
        + SEVERITY_WEIGHTS["shape"] * shape_score
        + SEVERITY_WEIGHTS["confidence"] * confidence_score
    )

    if severity_score < SEVERITY_THRESHOLDS["minor_max"]:
        severity = "Minor"
    elif severity_score < SEVERITY_THRESHOLDS["moderate_max"]:
        severity = "Moderate"
    else:
        severity = "Severe"

    if damage_class in ("Glass Shatter", "Lamp Broken") and severity == "Minor":
        severity = "Moderate"

    return {
        "severity": severity,
        "severity_score": round(severity_score, 3),
        "damage_percentage": round(damage_percentage, 2),
        "shape_irregularity": round(shape_score, 3),
    }


# ---------------------------------------------------------------------------
# Repair recommendations
# ---------------------------------------------------------------------------
REPAIR_RECOMMENDATIONS = {
    "Dent": {"Minor": ["Paintless Dent Repair (PDR)"], "Moderate": ["Panel Repair", "Filler + Repaint"], "Severe": ["Panel Replacement"]},
    "Scratch": {"Minor": ["Buffing", "Polishing"], "Moderate": ["Buffing", "Repainting"], "Severe": ["Repainting", "Panel Refinishing"]},
    "Crack": {"Minor": ["Sealant / Filler Repair"], "Moderate": ["Panel Repair", "Reinforcement"], "Severe": ["Replace Component"]},
    "Glass Shatter": {"Minor": ["Windshield Repair Kit"], "Moderate": ["Replace Windshield"], "Severe": ["Replace Windshield"]},
    "Lamp Broken": {"Minor": ["Replace Lamp Cover"], "Moderate": ["Replace Lamp"], "Severe": ["Replace Lamp Assembly"]},
    "Tire Flat": {"Minor": ["Tire Puncture Repair"], "Moderate": ["Tire Repair or Replacement (inspect first)"], "Severe": ["Replace Tire"]},
}


def get_repair_recommendation(damage_class, severity):
    class_rules = REPAIR_RECOMMENDATIONS.get(damage_class)
    if class_rules is None:
        return ["Manual inspection recommended"]
    return class_rules.get(severity, class_rules.get("Moderate", ["Manual inspection recommended"]))


# ---------------------------------------------------------------------------
# Full end-to-end assessment -- returns numpy images + a results dict
# (Flask app is responsible for turning the images into files/base64)
# ---------------------------------------------------------------------------
def run_assessment(image_path, apply_enhancement=True):
    if _unet_model is None or _clf_model is None:
        raise RuntimeError("Models are not loaded. Call load_models() first.")

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not read image from {image_path}")

    original_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    enhanced_img = enhance_image_adaptive(original_img, enable=apply_enhancement)

    seg_input = cv2.resize(enhanced_img, (SEG_SIZE, SEG_SIZE)).astype(np.float32) / 255.0
    mask_pred = _unet_model.predict(np.expand_dims(seg_input, 0), verbose=0)[0]
    binary_mask = (mask_pred.squeeze() > 0.5).astype(np.uint8)

    roi = extract_roi((seg_input * 255).astype(np.uint8), binary_mask)
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    clf_input = preprocess_input(roi_resized.astype(np.float32))

    probs = _clf_model.predict(np.expand_dims(clf_input, 0), verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    damage_class = CLASS_NAMES[pred_idx]
    confidence = float(probs[pred_idx])

    severity_info = assess_damage_severity(binary_mask, confidence, damage_class)
    severity = severity_info["severity"]

    heatmap, _ = make_gradcam_heatmap(
        np.expand_dims(clf_input, 0),
        _clf_model,
        _last_conv_layer_name,
        pred_index=pred_idx,
    )
    gradcam_img = overlay_heatmap(roi_resized, heatmap)

    recommendations = get_repair_recommendation(damage_class, severity)

    # Full-size mask overlay on the enhanced image, for display
    mask_disp = cv2.resize(binary_mask, (enhanced_img.shape[1], enhanced_img.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay_display = enhanced_img.copy()
    overlay_display[mask_disp > 0] = [255, 0, 0]
    overlay_display = cv2.addWeighted(enhanced_img, 0.6, overlay_display, 0.4, 0)

    images = {
        "original": original_img,
        "enhanced": enhanced_img,
        "mask": (mask_disp * 255).astype(np.uint8),
        "overlay": overlay_display,
        "gradcam": gradcam_img,
    }

    result = {
        "damage_class": damage_class,
        "confidence": confidence,
        "severity": severity,
        "severity_score": severity_info["severity_score"],
        "damage_percentage": severity_info["damage_percentage"],
        "shape_irregularity": severity_info["shape_irregularity"],
        "recommendations": recommendations,
        "all_probs": dict(zip(CLASS_NAMES, probs.tolist())),
    }

    return images, result
