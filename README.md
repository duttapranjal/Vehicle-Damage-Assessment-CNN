# Vehicle Damage Assessment — Flask Interface

A web interface for your notebook's pipeline: adaptive image enhancement →
U-Net damage segmentation → ROI extraction → EfficientNetB0 classification →
severity scoring → Grad-CAM explanation → repair recommendation.

## 1. Get your trained models onto your computer

In your Colab notebook, after training, download these two files
(they were saved under `CFG.MODEL_DIR`, e.g. `/content/cardd_project/models/`):

- `unet_final.keras`
- `efficientnet_classifier_final.keras`

You can download them from the Colab file browser (right-click → Download),
or run this in a Colab cell:

```python
from google.colab import files
files.download(f'{CFG.MODEL_DIR}/unet_final.keras')
files.download(f'{CFG.MODEL_DIR}/efficientnet_classifier_final.keras')
```

Place both files into this project's `models/` folder, so you have:

```
vehicle_damage_app/
  models/
    unet_final.keras
    efficientnet_classifier_final.keras
```

## 2. Install dependencies

It's best to use a virtual environment:

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Run the app

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

## 4. Using it

1. Upload a photo of vehicle damage (PNG/JPG).
2. Optionally toggle "Apply adaptive image enhancement".
3. Click "Analyze Damage".
4. You'll see: predicted damage type, confidence, severity, class
   probabilities, and the five pipeline stage images (original, enhanced,
   mask, overlay, Grad-CAM).

## How it's wired up

- `inference.py` — a direct port of your notebook's functions
  (`enhance_image_adaptive`, `extract_roi`, `make_gradcam_heatmap`,
  `assess_damage_severity`, `get_repair_recommendation`, etc.) plus
  `run_assessment()`, which mirrors your `run_full_assessment()`.
- `app.py` — the Flask routes: `/` shows the upload form, `/predict`
  handles the POST, runs `inference.run_assessment()`, saves the stage
  images to `static/results/`, and re-renders the page with results.
- `templates/index.html` — the single-page UI (form + results).
- `static/style.css` — styling.

## Notes / things you may want to extend

- Right now the PDF report generation (`generate_pdf_report` in your
  notebook) isn't wired into the web app — only the on-page results are
  shown. You could add a "Download PDF" button by importing that function
  into `inference.py` and adding a `/report/<run_id>` route.
- Uploaded images and result images accumulate in `static/uploads/` and
  `static/results/` — fine for a class project/demo, but you'd want a
  cleanup job or a database for anything longer-running.
- For a public demo, don't run with `debug=True` — that's for local
  development only.
