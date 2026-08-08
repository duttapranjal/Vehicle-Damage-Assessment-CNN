import io
import json
import os
import uuid
from typing import Dict, Tuple

import cv2
import numpy as np
from flask import Flask, flash, redirect, render_template, request, send_file, url_for
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

import inference

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
RESULT_FOLDER = os.path.join(BASE_DIR, "static", "results")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)
app.secret_key = "change-this-secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def _validate_uploaded_image(file_storage) -> Tuple[bytes, np.ndarray]:
    if file_storage.filename == "":
        raise ValueError("No file selected.")
    if not allowed_file(file_storage.filename):
        raise ValueError("Please upload a PNG or JPG image.")

    file_storage.stream.seek(0)
    image_bytes = file_storage.read()
    file_storage.stream.seek(0)

    if not image_bytes:
        raise ValueError("Uploaded file is empty.")

    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("The uploaded file is not a valid image or is corrupt.")
    if img.shape[0] < 16 or img.shape[1] < 16:
        raise ValueError("The uploaded image is too small for analysis.")
    return image_bytes, img


def _save_stage_images(run_id: str, images: Dict[str, np.ndarray]) -> Dict[str, str]:
    saved = {}
    for key, img in images.items():
        out_name = f"{run_id}_{key}.png"
        out_path = os.path.join(RESULT_FOLDER, out_name)
        if img is None:
            continue
        if img.ndim == 2:
            cv2.imwrite(out_path, img)
        else:
            cv2.imwrite(out_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        saved[key] = f"results/{out_name}"
    return saved


def _write_run_metadata(run_id: str, uploaded_filename: str, saved_images: Dict[str, str], result: Dict) -> None:
    metadata_path = os.path.join(RESULT_FOLDER, f"{run_id}.json")
    metadata = {
        "run_id": run_id,
        "uploaded_filename": uploaded_filename,
        "result": result,
        "images": {},
    }
    for key, rel_path in saved_images.items():
        metadata["images"][key] = {
            "url": rel_path,
            "path": os.path.join(BASE_DIR, "static", rel_path),
        }
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def generate_pdf_report(metadata: Dict) -> bytes:
    from textwrap import wrap

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    pdf.setTitle(f"Vehicle Damage Report - {metadata['run_id']}")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(40, height - 50, "Vehicle Damage Assessment Report")

    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, height - 75, f"Run ID: {metadata['run_id']}")
    pdf.drawString(40, height - 95, f"Uploaded Image: {metadata['uploaded_filename']}")

    result = metadata.get("result", {})
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(40, height - 125, "Prediction Summary")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, height - 145, f"Damage Type: {result.get('damage_class', 'Unknown')}")
    pdf.drawString(40, height - 165, f"Confidence: {result.get('confidence', 0.0):.1%}")
    pdf.drawString(40, height - 185, f"Severity: {result.get('severity', 'Unknown')}")
    pdf.drawString(40, height - 205, f"Damaged Area: {result.get('damage_percentage', 0.0)}%")

    recommendations = result.get("recommendations", [])
    rec_text = "; ".join(recommendations) if recommendations else "No recommendation available"
    rec_lines = wrap(rec_text, width=90)
    pdf.drawString(40, height - 235, "Recommendation:")
    y_pos = height - 255
    for line in rec_lines:
        pdf.drawString(55, y_pos, line)
        y_pos -= 14

    image_paths = []
    for key in ("original", "enhanced", "overlay"):
        image_info = metadata.get("images", {}).get(key, {})
        if image_info.get("path") and os.path.exists(image_info["path"]):
            image_paths.append(image_info["path"])

    if image_paths:
        x_positions = [40, 210, 380]
        y_top = height - 420
        for idx, image_path in enumerate(image_paths[:3]):
            x = x_positions[idx]
            if os.path.exists(image_path):
                pdf.drawImage(image_path, x, y_top, width=120, height=90, preserveAspectRatio=True)

    pdf.showPage()
    pdf.save()
    buffer.seek(0)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("index"))

    file = request.files["image"]
    try:
        _, _ = _validate_uploaded_image(file)
    except ValueError as exc:
        flash(str(exc))
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Please upload a PNG or JPG image.")
        return redirect(url_for("index"))

    run_id = uuid.uuid4().hex[:10]
    ext = file.filename.rsplit(".", 1)[1].lower()
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], f"{run_id}.{ext}")
    file.save(upload_path)

    apply_enhancement = request.form.get("apply_enhancement") == "on"

    try:
        images, result = inference.run_assessment(upload_path, apply_enhancement=apply_enhancement)
    except FileNotFoundError as exc:
        flash(str(exc))
        return redirect(url_for("index"))
    except ValueError as exc:
        flash(f"Image processing failed: {exc}")
        return redirect(url_for("index"))
    except Exception as exc:  # pragma: no cover - defensive fallback
        flash(f"Something went wrong while processing the image: {exc}")
        return redirect(url_for("index"))

    saved_images = _save_stage_images(run_id, images)
    result["run_id"] = run_id
    _write_run_metadata(run_id, file.filename, saved_images, result)

    return render_template("index.html", result=result, images=saved_images, run_id=run_id)


@app.route("/demo/<damage_type>", methods=["GET"])
def demo(damage_type: str):
    # Build demo image path
    demo_path = os.path.join(BASE_DIR, "assets", "demo_images", f"{damage_type}.jpg")
    if not os.path.exists(demo_path):
        flash("Demo image not available.")
        return redirect(url_for("index"))

    run_id = f"demo_{damage_type}"
    try:
        images, result = inference.run_assessment(demo_path, apply_enhancement=True)
    except FileNotFoundError as exc:
        flash(str(exc))
        return redirect(url_for("index"))
    except ValueError as exc:
        flash(f"Image processing failed: {exc}")
        return redirect(url_for("index"))
    except Exception as exc:  # pragma: no cover
        flash(f"Something went wrong while processing the demo image: {exc}")
        return redirect(url_for("index"))

    saved_images = _save_stage_images(run_id, images)
    result["run_id"] = run_id
    _write_run_metadata(run_id, os.path.basename(demo_path), saved_images, result)

    return render_template("index.html", result=result, images=saved_images, run_id=run_id)


@app.route("/report/<run_id>")
def report(run_id: str):
    metadata_path = os.path.join(RESULT_FOLDER, f"{run_id}.json")
    if not os.path.exists(metadata_path):
        flash("No report is available for that run.")
        return redirect(url_for("index"))

    with open(metadata_path, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    pdf_bytes = generate_pdf_report(metadata)
    return send_file(
        io.BytesIO(pdf_bytes),
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"damage_report_{run_id}.pdf",
    )


if __name__ == "__main__":
    print("Loading models... this can take a moment.")
    inference.load_models()
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
