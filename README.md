# 🚗 AI Vehicle Damage Assessment

> An end-to-end deep learning system that detects vehicle damage type, severity, and repair recommendations from a single image — built on U-Net segmentation, EfficientNetB0 classification, and Grad-CAM explainability.

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit)](https://vehicle-damage-assessment-cnn-dpfyy5nvrbxnrcgfjf3uiv.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-duttapranjal-181717?logo=github)](https://github.com/duttapranjal/Vehicle-Damage-Assessment-CNN)
[![Models](https://img.shields.io/badge/Models-HuggingFace-FFD21F?logo=huggingface)](https://huggingface.co/Pranjaldutta129/Vehicle-Damage-Models)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.19-FF6F00?logo=tensorflow)](https://tensorflow.org)

---

## 📸 Demo

<!-- Add a GIF or screenshot of the running app here after deployment -->
<!-- Drag an image into this section or use: ![Demo](assets/demo.gif) -->

Upload any vehicle image → get damage type, severity badge, Grad-CAM heatmap, and a downloadable PDF report in seconds.

---

## 🏗️ Pipeline Architecture

```
Input Image
    │
    ▼
Adaptive Enhancement        ← CLAHE + gamma correction + bilateral filter
    │
    ▼
U-Net Segmentation          ← Binary mask of damaged region
    │                          Dice: TBD  |  IoU: TBD
    ▼
ROI Extraction              ← Crops and isolates the damage area
    │
    ▼
EfficientNetB0 Classifier   ← 6 damage classes
    │                          Accuracy: TBD%  |  F1: TBD
    ▼
Grad-CAM Explainability     ← Visual attention heatmap
    │
    ▼
Severity Scoring            ← Weighted: area (60%) + shape (25%) + confidence (15%)
    │
    ▼
PDF Report Generation       ← Downloadable assessment report
```

---

## 📊 Results

| Model | Accuracy | F1 (macro) | Seg. Dice | Seg. IoU |
|-------|----------|------------|-----------|----------|
| Baseline (full image, EfficientNetB0) | TBD% | TBD | — | — |
| **Ours (U-Net ROI + EfficientNetB0)** | **TBD%** | **TBD** | TBD | TBD |

*Fill in real numbers after running the training notebook.*

---

## 🏷️ Damage Classes

| Class | Description |
|-------|-------------|
| 🔨 Dent | Panel deformation without paint damage |
| ✏️ Scratch | Surface paint damage |
| 💔 Crack | Structural fractures |
| 🪟 Glass Shatter | Windshield or window damage |
| 💡 Lamp Broken | Headlight or tail light damage |
| 🔴 Tire Flat | Tyre puncture or blowout |

---

## ⚙️ Notable Engineering Decisions

**Keras 3 / TF 2.19 flat-graph fix**
Nested Functional sub-graphs break Grad-CAM's `GradientTape` in TF 2.19+. Fixed by flattening into a single `Model(inputs=base_model.input, outputs=outputs)` call — avoids the sub-model boundary that blocks gradient flow.

**3-stage progressive fine-tuning**
Frozen head → unfreeze 30 layers → unfreeze 80 layers, each at a lower LR (`1e-3 → 1e-4 → 1e-5`). Separate callbacks per stage so checkpoints don't overwrite each other.

**BCE + Dice combined loss**
Used `bce_dice_loss` instead of vanilla BCE for U-Net. Dice loss handles class imbalance in sparse binary masks far better than pixel accuracy.

**Multi-factor severity scoring**
```python
severity_score = 0.60 * area_score + 0.25 * shape_score + 0.15 * confidence_score
```
Domain overrides: Glass Shatter and Lamp Broken are never classified as Minor.

**Sequential model loading with GC**
On memory-constrained servers, `gc.collect()` + `tf.keras.backend.clear_session()` between U-Net and EfficientNet loads prevents OOM crashes.

---

## 🗂️ Project Structure

```
Vehicle-Damage-Assessment-CNN/
├── streamlit_app.py          ← Streamlit UI entry point
├── inference.py              ← Full ML pipeline (enhancement → segmentation → classification → Grad-CAM → severity)
├── requirements.txt          ← Python dependencies
├── packages.txt              ← System dependencies for Streamlit Cloud
├── .streamlit/
│   └── config.toml           ← Dark theme configuration
├── notebooks/
│   └── Vehicle_Damage_Assessment_BootCamp_Project_Final.ipynb
├── models/                   ← Weights downloaded from HuggingFace Hub at runtime
├── assets/
│   └── demo_images/          ← Demo images for in-app testing
└── tests/
    └── test_inference.py     ← Pytest unit tests
```

---

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/duttapranjal/Vehicle-Damage-Assessment-CNN.git
cd Vehicle-Damage-Assessment-CNN

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_REPO_ID="Pranjaldutta129/Vehicle-Damage-Models"
export HF_TOKEN="your_huggingface_token"

# Run the app
streamlit run streamlit_app.py
```

Or create `.streamlit/secrets.toml`:
```toml
HF_REPO_ID = "Pranjaldutta129/Vehicle-Damage-Models"
HF_TOKEN = "your_huggingface_token"
```

---

## 🧰 Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit |
| Segmentation | U-Net (custom, trained from scratch) |
| Classification | EfficientNetB0 (3-stage progressive fine-tuning) |
| Explainability | Grad-CAM |
| Augmentation | Albumentations |
| Report | ReportLab PDF |
| Dataset | CarDD (Car Damage Detection Dataset) |
| Model Hosting | HuggingFace Hub |
| Deployment | Streamlit Community Cloud |

---

## 📦 Deployment

Hosted on **Streamlit Community Cloud** — free, permanent URL, no card required.
Model weights (~300MB total) are hosted on **HuggingFace Hub** and downloaded at app startup.

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://vehicle-damage-assessment-cnn-dpfyy5nvrbxnrcgfjf3uiv.streamlit.app/)

---

## 🧪 Tests

```bash
pytest tests/ -v
```

---

## 👤 Author

**Pranjal Dutta**
B.Tech Computer Science | Data Science enthusiast

[![GitHub](https://img.shields.io/badge/GitHub-duttapranjal-181717?logo=github)](https://github.com/duttapranjal)

---

## 📄 License

MIT License — feel free to use, modify, and distribute with attribution.
