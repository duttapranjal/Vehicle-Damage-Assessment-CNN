[![Live Demo](https://img.shields.io/badge/Live%20Demo-Click%20Here-brightgreen)](https://YOUR_RENDER_URL_HERE)
[![CI](https://github.com/duttapranjal/Vehicle-Damage-Assessment-CNN/actions/workflows/ci.yml/badge.svg)](https://github.com/duttapranjal/Vehicle-Damage-Assessment-CNN/actions/workflows/ci.yml)

# Vehicle-Damage-Assessment-CNN

Deep Learning project for vehicle damage assessment using CNN, image
segmentation, and Grad-CAM. This repository contains a Flask web
interface that mirrors the notebook pipeline: adaptive image enhancement,
U-Net damage segmentation, ROI extraction, EfficientNet classification,
severity scoring, Grad-CAM explanation, and repair recommendation.

## Project Overview

This project performs vehicle damage assessment using Computer Vision and
Deep Learning. The web app bundles the inference pipeline so you can
upload an image and receive segmentation, classification, severity, and
visual explanations.

## Results
| Model | Accuracy | F1 (macro) | Seg. Dice | Seg. IoU |
|-------|----------|-----------|-----------|----------|
| Baseline (full image) | TBD% | TBD | — | — |
| **Ours (ROI-based)** | **TBD%** | **TBD** | TBD | TBD |

## Screenshots
<!-- Add 2-3 screenshots of the running web app here after deployment -->
<!-- Drag images into this edit window or use: ![alt](assets/screenshot1.png) -->

## Features

- Image preprocessing
- Data augmentation
- CNN-based classification
- Transfer Learning
- Model evaluation
- Grad-CAM visualization
- Confusion Matrix
- Accuracy and Loss graphs

## Quickstart — Run the Flask app locally

1. Place the trained model files into the `models/` folder:

   - `unet_final.keras`
   - `efficientnet_classifier_final.keras`

   Example layout:

   ```
   Vehicle_damage_app/
     models/
       unet_final.keras
       efficientnet_classifier_final.keras
   ```

2. Create and activate a virtual environment, then install dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. Run the app:

   ```bash
   python app.py
   ```

   Open http://127.0.0.1:5000 in your browser.

## Using the Web App

1. Upload a photo of vehicle damage (PNG/JPG).
2. Optionally toggle adaptive image enhancement.
3. Click "Analyze Damage" and view predictions, masks, overlays, and
   Grad-CAM.

## How it's wired up

- `inference.py` — ports notebook functions (`enhance_image_adaptive`,
  `extract_roi`, `make_gradcam_heatmap`, `assess_damage_severity`,
  `get_repair_recommendation`) and exposes `run_assessment()` used by the
  web app.
- `app.py` — Flask routes: `/` (upload form) and `/predict` (runs the
  pipeline and serves results saved under `static/results/`).
- `templates/index.html` — UI (form + results).
- `static/` — styles and generated result images.

## Notes & Next Steps

- PDF report generation is available at `/report/<run_id>` after any prediction.
- Uploaded images and results accumulate in `static/uploads/` and
  `static/results/` — consider a cleanup job for long-running use.
- Production deployment uses gunicorn via `render.yaml`. `debug=False` is enforced.

## Deployment
Hosted on Render. Models served from HuggingFace Hub.
See render.yaml for the deployment configuration.

## Author

Pranjal Dutta

