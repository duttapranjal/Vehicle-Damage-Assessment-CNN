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

import gc
import logging
import os
import sys

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input

try:
    from huggingface_hub import hf_hub_download
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

try:
    import mlflow
    HAS_MLFLOW = os.environ.get("ENABLE_MLFLOW", "false").lower() == "true"
except ImportError:
    HAS_MLFLOW = False

# ---------------------------------------------------------------------------
# Config -- must match the notebook
# ---------------------------------------------------------------------------
IMG_SIZE = 224
SEG_SIZE = 256
CLASS_NAMES = ["Dent", "Scratch", "Crack", "Glass Shatter", "Lamp Broken", "Tire Flat"]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

UNET_PATH = os.path.join(MODEL_DIR, "unet_final.keras")
CLF_PATH  = os.path.join(MODEL_DIR, "efficientnet_classifier_final.keras")

try:
    import streamlit as _st
    HF_REPO_ID = _st.secrets.get("HF_REPO_ID", "Pranjaldutta129/Vehicle-Damage-Models")
    HF_TOKEN   = _st.secrets.get("HF_TOKEN",   os.environ.get("HF_TOKEN", None))
except Exception:
    HF_REPO_ID = os.environ.get("HF_REPO_ID", "Pranjaldutta129/Vehicle-Damage-Models")
    HF_TOKEN   = os.environ.get("HF_TOKEN", None)

# ---------------------------------------------------------------------------
# Global model handles — loaded once at startup
# ---------------------------------------------------------------------------
_unet_model          = None
_clf_model           = None
_last_conv_layer_name = None


# ---------------------------------------------------------------------------
# Custom loss / metric — must be registered before load_model()
# ---------------------------------------------------------------------------
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
    bce  = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
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


_ORIGINAL_DENSE_INIT = tf.keras.layers.Dense.__init__


def _patch_keras_dense_deserialization() -> None:
    def _compat_dense_init(self, *args, **kwargs):
        kwargs.pop("quantization_config", None)
        return _ORIGINAL_DENSE_INIT(self, *args, **kwargs)
    tf.keras.layers.Dense.__init__ = _compat_dense_init


# ---------------------------------------------------------------------------
# Model loading — called ONCE at app startup (not on first request)
# FIX 3: gc.collect() between the two model loads to free RAM on free tier
# FIX 2: models are loaded here at startup so gunicorn --preload triggers
#         this before any worker is forked, avoiding the 60s worker timeout
# ---------------------------------------------------------------------------
def load_models():
    """
    Loads both .keras models into memory sequentially.
    Called once at app startup via gunicorn --preload.

    Key changes vs previous version:
    - gc.collect() between U-Net and EfficientNet loads (Fix 3)
      Forces Python to free any temporary allocations from U-Net loading
      before the larger EfficientNet model is pulled into RAM.
    - Sequential download: U-Net is fully downloaded AND loaded before
      EfficientNet download begins, halving peak memory during loading.
    - Explicit del of intermediate tensors after each load.
    """
    global _unet_model, _clf_model, _last_conv_layer_name

    logger.info("=== Starting model loading ===")

    os.makedirs(MODEL_DIR, exist_ok=True)
    _patch_keras_dense_deserialization()

    # ── helper: local path first, HF Hub fallback ──────────────────────────
    def _resolve(local_path: str, hf_filename: str) -> str:
        if os.path.exists(local_path):
            logger.info(f"Using local model: {local_path}")
            return local_path
        if HF_AVAILABLE and HF_REPO_ID:
            logger.info(f"Downloading {hf_filename} from HuggingFace Hub ...")
            cache = os.path.join(BASE_DIR, ".hf_cache")
            return hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=hf_filename,
                token=HF_TOKEN,
                cache_dir=cache,
            )
        raise FileNotFoundError(
            f"Model not found at {local_path} and HF_REPO_ID env var is not set."
        )

    try:
        # ── STEP 1: Resolve paths (downloads happen here if needed) ────────
        logger.info("Resolving U-Net cpath ...")
        unet_path = _resolve(UNET_PATH, "unet_final.keras")

        logger.info("Resolving EfficientNet path ...")
        clf_path = _resolve(CLF_PATH, "efficientnet_classifier_final.keras")

        # ── STEP 2: Load U-Net ──────────────────────────────────────────────
        logger.info("Loading U-Net model ...")
        _unet_model = tf.keras.models.load_model(
            unet_path,
            custom_objects={
                "bce_dice_loss":    bce_dice_loss,
                "dice_coefficient": dice_coefficient,
            },
            compile=False,
        )
        logger.info(f"✓ U-Net loaded  |  output shape: {_unet_model.output_shape}")

        # ── FIX 3: force garbage collection before loading second model ─────
        # On Render's free tier (512MB RAM) loading both models back-to-back
        # without GC causes peak usage to spike and triggers SIGKILL.
        # gc.collect() + clearing the TF graph cache frees temporary tensors
        # that accumulated during U-Net deserialisation before EfficientNet
        # is pulled into memory.
        gc.collect()
        tf.keras.backend.clear_session()   # clears Keras graph cache
        gc.collect()                       # second pass catches circular refs
        logger.info("GC complete — RAM freed before EfficientNet load")

        # ── STEP 3: Load EfficientNet classifier ───────────────────────────
        logger.info("Loading EfficientNet classifier ...")
        _clf_model = tf.keras.models.load_model(clf_path, compile=False)
        logger.info(f"✓ EfficientNet loaded  |  output shape: {_clf_model.output_shape}")

        # ── STEP 4: Resolve Grad-CAM layer name ────────────────────────────
        _last_conv_layer_name = _find_last_conv_layer_name(_clf_model)
        logger.info(f"✓ Grad-CAM layer: {_last_conv_layer_name}")

        # Final GC after everything is loaded
        gc.collect()
        logger.info("=== All models ready ===")

    except Exception as e:
        logger.error(f"Model loading failed: {e}", exc_info=True)
        raise


def models_loaded() -> bool:
    """Returns True if both models are in memory. Used by /health endpoint."""
    return _unet_model is not None and _clf_model is not None


# ---------------------------------------------------------------------------
# Image enhancement (adaptive)
# ---------------------------------------------------------------------------
def apply_clahe(img_rgb):
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_eq  = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((l_eq, a, b)), cv2.COLOR_LAB2RGB)


def apply_bilateral_filter(img_rgb):
    return cv2.bilateralFilter(img_rgb, d=9, sigmaColor=75, sigmaSpace=75)


def apply_contrast_enhancement(img_rgb, alpha=1.3, beta=10):
    return cv2.convertScaleAbs(img_rgb, alpha=alpha, beta=beta)


def gamma_correction(img_rgb, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)]
    ).astype(np.uint8)
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
# ROI extraction
# ---------------------------------------------------------------------------
def extract_roi(img_rgb, binary_mask, pad_ratio=0.1):
    ys, xs = np.where(binary_mask > 0)
    if len(xs) == 0:
        return img_rgb

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h, w   = binary_mask.shape

    pad_x = int((x2 - x1) * pad_ratio)
    pad_y = int((y2 - y1) * pad_ratio)
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

    grads       = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap      = tf.squeeze(heatmap)
    heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    final_idx = pred_index.numpy() if hasattr(pred_index, "numpy") else pred_index
    return heatmap.numpy(), int(final_idx)


def overlay_heatmap(img_rgb_uint8, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (img_rgb_uint8.shape[1], img_rgb_uint8.shape[0]))
    heatmap_color   = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color   = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(img_rgb_uint8, 1 - alpha, heatmap_color, alpha, 0)


# ---------------------------------------------------------------------------
# Severity scoring
# ---------------------------------------------------------------------------
SEVERITY_THRESHOLDS = {"minor_max": 0.33, "moderate_max": 0.66}
SEVERITY_WEIGHTS    = {"area": 0.60, "shape": 0.25, "confidence": 0.15}


def compute_shape_irregularity(binary_mask):
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return 0.0
    largest  = max(contours, key=cv2.contourArea)
    area     = cv2.contourArea(largest)
    if area < 1:
        return 0.0
    hull      = cv2.convexHull(largest)
    hull_area = cv2.contourArea(hull)
    if hull_area < 1:
        return 0.0
    solidity = area / hull_area
    return float(np.clip(1 - solidity, 0, 1))


def assess_damage_severity(binary_mask, confidence, damage_class=None):
    total_px        = binary_mask.size
    damage_px       = int(binary_mask.sum())
    damage_pct      = (damage_px / total_px) * 100 if total_px > 0 else 0.0

    area_score       = float(np.clip(damage_pct / 25.0, 0, 1))
    shape_score      = compute_shape_irregularity(binary_mask)
    confidence_score = float(np.clip(confidence, 0, 1))

    severity_score = (
        SEVERITY_WEIGHTS["area"]       * area_score
        + SEVERITY_WEIGHTS["shape"]      * shape_score
        + SEVERITY_WEIGHTS["confidence"] * confidence_score
    )

    if severity_score < SEVERITY_THRESHOLDS["minor_max"]:
        severity = "Minor"
    elif severity_score < SEVERITY_THRESHOLDS["moderate_max"]:
        severity = "Moderate"
    else:
        severity = "Severe"

    # Domain overrides — Glass Shatter and Lamp Broken are never Minor
    if damage_class in ("Glass Shatter", "Lamp Broken") and severity == "Minor":
        severity = "Moderate"

    return {
        "severity":          severity,
        "severity_score":    round(severity_score, 3),
        "damage_percentage": round(damage_pct, 2),
        "shape_irregularity": round(shape_score, 3),
    }


# ---------------------------------------------------------------------------
# Repair recommendations
# ---------------------------------------------------------------------------
REPAIR_RECOMMENDATIONS = {
    "Dent":          {"Minor": ["Paintless Dent Repair (PDR)"],
                      "Moderate": ["Panel Repair", "Filler + Repaint"],
                      "Severe":   ["Panel Replacement"]},
    "Scratch":       {"Minor": ["Buffing", "Polishing"],
                      "Moderate": ["Buffing", "Repainting"],
                      "Severe":   ["Repainting", "Panel Refinishing"]},
    "Crack":         {"Minor": ["Sealant / Filler Repair"],
                      "Moderate": ["Panel Repair", "Reinforcement"],
                      "Severe":   ["Replace Component"]},
    "Glass Shatter": {"Minor": ["Windshield Repair Kit"],
                      "Moderate": ["Replace Windshield"],
                      "Severe":   ["Replace Windshield"]},
    "Lamp Broken":   {"Minor": ["Replace Lamp Cover"],
                      "Moderate": ["Replace Lamp"],
                      "Severe":   ["Replace Lamp Assembly"]},
    "Tire Flat":     {"Minor": ["Tire Puncture Repair"],
                      "Moderate": ["Tire Repair or Replacement (inspect first)"],
                      "Severe":   ["Replace Tire"]},
}


def get_repair_recommendation(damage_class, severity):
    class_rules = REPAIR_RECOMMENDATIONS.get(damage_class)
    if class_rules is None:
        return ["Manual inspection recommended"]
    return class_rules.get(severity, class_rules.get("Moderate", ["Manual inspection recommended"]))


# ---------------------------------------------------------------------------
# Full end-to-end assessment pipeline
# ---------------------------------------------------------------------------
def run_assessment(image_path, apply_enhancement=True):
    if not models_loaded():
        raise RuntimeError("Models are not loaded. Call load_models() first.")

    logger.info(f"Starting assessment for: {image_path}")

    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f"Could not read image from {image_path}")

    original_img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    enhanced_img = enhance_image_adaptive(original_img, enable=apply_enhancement)

    # Segmentation
    logger.info("Running U-Net segmentation ...")
    seg_input = cv2.resize(enhanced_img, (SEG_SIZE, SEG_SIZE)).astype(np.float32) / 255.0
    mask_pred  = _unet_model.predict(np.expand_dims(seg_input, 0), verbose=0)[0]
    binary_mask = (mask_pred.squeeze() > 0.5).astype(np.uint8)
    logger.info(f"Segmentation done — damaged pixels: {binary_mask.sum()}")

    # Classification
    logger.info("Extracting ROI and classifying ...")
    roi         = extract_roi((seg_input * 255).astype(np.uint8), binary_mask)
    roi_resized = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    clf_input   = preprocess_input(roi_resized.astype(np.float32))

    probs       = _clf_model.predict(np.expand_dims(clf_input, 0), verbose=0)[0]
    pred_idx    = int(np.argmax(probs))
    damage_class = CLASS_NAMES[pred_idx]
    confidence  = float(probs[pred_idx])
    logger.info(f"Classification: {damage_class} ({confidence:.1%})")

    # Severity
    severity_info = assess_damage_severity(binary_mask, confidence, damage_class)
    severity      = severity_info["severity"]

    # Grad-CAM
    logger.info("Generating Grad-CAM ...")
    heatmap, _ = make_gradcam_heatmap(
        np.expand_dims(clf_input, 0),
        _clf_model,
        _last_conv_layer_name,
        pred_index=pred_idx,
    )
    gradcam_img = overlay_heatmap(roi_resized, heatmap)

    recommendations = get_repair_recommendation(damage_class, severity)

    # Full-size mask overlay for display
    mask_disp = cv2.resize(
        binary_mask,
        (enhanced_img.shape[1], enhanced_img.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )
    overlay_display = enhanced_img.copy()
    overlay_display[mask_disp > 0] = [255, 0, 0]
    overlay_display = cv2.addWeighted(enhanced_img, 0.6, overlay_display, 0.4, 0)

    images = {
        "original": original_img,
        "enhanced": enhanced_img,
        "mask":     (mask_disp * 255).astype(np.uint8),
        "overlay":  overlay_display,
        "gradcam":  gradcam_img,
    }

    result = {
        "damage_class":       damage_class,
        "confidence":         confidence,
        "severity":           severity,
        "severity_score":     severity_info["severity_score"],
        "damage_percentage":  severity_info["damage_percentage"],
        "shape_irregularity": severity_info["shape_irregularity"],
        "recommendations":    recommendations,
        "all_probs":          dict(zip(CLASS_NAMES, probs.tolist())),
    }

    if HAS_MLFLOW:
        with mlflow.start_run(run_name="web_inference", nested=True):
            mlflow.log_params({
                "damage_class":       result["damage_class"],
                "severity":           result["severity"],
                "apply_enhancement":  apply_enhancement,
            })
            mlflow.log_metrics({
                "confidence":        result["confidence"],
                "damage_percentage": result["damage_percentage"],
                "severity_score":    result["severity_score"],
            })

    logger.info(f"✓ Assessment complete: {damage_class} | {severity}")
    return images, result