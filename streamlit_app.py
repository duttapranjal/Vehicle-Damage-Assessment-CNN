"""
streamlit_app.py
----------------
Streamlit replacement for the Flask app.py.
Runs the same inference pipeline (inference.py is unchanged).
Deploy to Streamlit Community Cloud — free, no card, no RAM fights.
"""

import io
import os
import sys
import tempfile

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# ── Page config — must be first Streamlit call ─────────────────────────────
st.set_page_config(
    page_title="AI Vehicle Damage Assessment",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS — dark tech aesthetic matching your existing Flask UI ────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0a0a0f;
    color: #e2e8f0;
}
.stApp { background-color: #0a0a0f; }

/* Header */
.hero-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2rem;
    font-weight: 500;
    color: #f0b429;
    letter-spacing: 0.08em;
    text-align: center;
    margin-bottom: 0.25rem;
}
.hero-sub {
    font-size: 0.85rem;
    color: #64748b;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.1em;
    margin-bottom: 2rem;
}

/* Metric cards */
.metric-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 1.5rem 0;
}
.metric-card {
    background: #12121a;
    border: 1px solid #1e1e2e;
    border-radius: 8px;
    padding: 16px;
    text-align: center;
}
.metric-label {
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
}
.metric-value {
    font-size: 1.4rem;
    font-weight: 600;
    color: #f0b429;
    font-family: 'JetBrains Mono', monospace;
}
.metric-value.green { color: #22c55e; }
.metric-value.red   { color: #ef4444; }
.metric-value.amber { color: #f0b429; }

/* Severity badge */
.severity-minor    { background:#14532d; color:#4ade80; padding:4px 14px; border-radius:20px; font-size:0.85rem; font-weight:600; }
.severity-moderate { background:#713f12; color:#fbbf24; padding:4px 14px; border-radius:20px; font-size:0.85rem; font-weight:600; }
.severity-severe   { background:#7f1d1d; color:#f87171; padding:4px 14px; border-radius:20px; font-size:0.85rem; font-weight:600; }

/* Pipeline step labels */
.step-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    text-align: center;
    margin-top: 6px;
}

/* Upload zone */
[data-testid="stFileUploader"] {
    border: 1.5px dashed #1e293b;
    border-radius: 12px;
    padding: 1rem;
    background: #0d0d18;
}

/* Buttons */
.stButton > button {
    background: #f0b429;
    color: #0a0a0f;
    font-weight: 600;
    border: none;
    border-radius: 6px;
    font-family: 'JetBrains Mono', monospace;
    letter-spacing: 0.05em;
    width: 100%;
}
.stButton > button:hover {
    background: #d4a017;
    color: #0a0a0f;
}

/* Progress bar colour */
.stProgress > div > div > div { background-color: #f0b429; }

/* Recommendation list */
.rec-item {
    background: #12121a;
    border-left: 3px solid #f0b429;
    border-radius: 0 6px 6px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    font-size: 0.9rem;
    color: #cbd5e1;
}

/* Prob bar */
.prob-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
    font-size: 0.82rem;
}
.prob-name  { width: 120px; color: #94a3b8; font-family:'JetBrains Mono',monospace; }
.prob-bar   { flex: 1; background:#1e1e2e; border-radius:4px; height:8px; overflow:hidden; }
.prob-fill  { height: 100%; background: #f0b429; border-radius: 4px; }
.prob-pct   { width: 44px; text-align:right; color:#f0b429; font-family:'JetBrains Mono',monospace; }
</style>
""", unsafe_allow_html=True)

# ── Import inference (unchanged from Flask version) ────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
import inference


# ── Load models once — cached for the session ──────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """
    st.cache_resource means this runs ONCE per server process and is shared
    across all user sessions. Equivalent to gunicorn --preload but simpler.
    """
    inference.load_models()
    return True


# ── PDF generation (same logic as app.py) ─────────────────────────────────
def generate_pdf(result, images_dict):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from textwrap import wrap

    buf = io.BytesIO()
    pdf = canvas.Canvas(buf, pagesize=letter)
    w, h = letter

    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(40, h - 50, "Vehicle Damage Assessment Report")
    pdf.setFont("Helvetica", 11)
    pdf.drawString(40, h - 80, f"Damage Type:  {result.get('damage_class', 'Unknown')}")
    pdf.drawString(40, h - 100, f"Confidence:   {result.get('confidence', 0):.1%}")
    pdf.drawString(40, h - 120, f"Severity:     {result.get('severity', 'Unknown')}")
    pdf.drawString(40, h - 140, f"Damaged Area: {result.get('damage_percentage', 0)}%")
    pdf.drawString(40, h - 160, f"Severity Score: {result.get('severity_score', 0):.3f}")

    recs = result.get("recommendations", [])
    rec_text  = "; ".join(recs) if recs else "No recommendation available"
    rec_lines = wrap(rec_text, width=90)
    pdf.drawString(40, h - 190, "Recommendation:")
    y = h - 210
    for line in rec_lines:
        pdf.drawString(55, y, line)
        y -= 14

    # Embed original + overlay + gradcam images
    labels = ["original", "overlay", "gradcam"]
    x_pos  = [40, 210, 380]
    y_top  = h - 400
    for idx, key in enumerate(labels):
        arr = images_dict.get(key)
        if arr is None:
            continue
        pil = Image.fromarray(arr.astype(np.uint8))
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            pil.save(tmp.name)
            pdf.drawImage(tmp.name, x_pos[idx], y_top,
                          width=130, height=98, preserveAspectRatio=True)
            os.unlink(tmp.name)
        pdf.setFont("Helvetica", 8)
        pdf.drawString(x_pos[idx], y_top - 12, key.upper())

    pdf.showPage()
    pdf.save()
    buf.seek(0)
    return buf.getvalue()


# ── Helper: numpy RGB → PIL ────────────────────────────────────────────────
def to_pil(arr):
    if arr is None:
        return None
    return Image.fromarray(arr.astype(np.uint8))


# ── Severity badge HTML ────────────────────────────────────────────────────
def severity_badge(severity):
    cls = {"Minor": "severity-minor",
           "Moderate": "severity-moderate",
           "Severe": "severity-severe"}.get(severity, "severity-moderate")
    return f'<span class="{cls}">{severity}</span>'


# ══════════════════════════════════════════════════════════════════════════════
# MAIN UI
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="hero-title">⬡ AI VEHICLE DAMAGE ASSESSMENT</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">enhance → segment → classify → grad-cam → severity</div>', unsafe_allow_html=True)

# ── Load models with a visible spinner ────────────────────────────────────
with st.spinner("Loading AI models... (first load ~60s on free tier)"):
    try:
        load_models()
    except Exception as e:
        st.error(f"Failed to load models: {e}")
        st.stop()

# ── Two-column layout: upload left, controls right ────────────────────────
col_upload, col_controls = st.columns([2, 1])

with col_upload:
    uploaded_file = st.file_uploader(
        "Upload a vehicle image",
        type=["jpg", "jpeg", "png"],
        help="Max 16MB. JPG or PNG.",
        label_visibility="collapsed",
    )

    # Demo image buttons
    st.markdown("**Or try a demo:**")
    d1, d2, d3 = st.columns(3)
    demo_choice = None
    with d1:
        if st.button("🚗 Dent"):
            demo_choice = "dent"
    with d2:
        if st.button("✏️ Scratch"):
            demo_choice = "scratch"
    with d3:
        if st.button("🪟 Glass"):
            demo_choice = "glass_shatter"

with col_controls:
    st.markdown("**Options**")
    apply_enhancement = st.toggle("Image enhancement", value=True,
        help="CLAHE + gamma + bilateral filter. Improves low-contrast images.")
    run_btn = st.button("▶ Run Assessment", use_container_width=True)

st.divider()

# ── Resolve input: uploaded file or demo ──────────────────────────────────
img_rgb   = None
img_label = None

if demo_choice:
    demo_path = os.path.join(BASE_DIR, "assets", "demo_images", f"{demo_choice}.jpg")
    if os.path.exists(demo_path):
        img_bgr   = cv2.imread(demo_path)
        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_label = f"demo_{demo_choice}.jpg"
        st.info(f"Demo image loaded: **{demo_choice.replace('_', ' ').title()}**")
    else:
        st.warning(f"Demo image not found at `assets/demo_images/{demo_choice}.jpg`. "
                    "Add demo images to that folder.")

elif uploaded_file:
    arr     = np.frombuffer(uploaded_file.read(), np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        st.error("Could not read image. Please upload a valid JPG or PNG.")
        st.stop()
    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_label = uploaded_file.name

# ── Show preview before running ───────────────────────────────────────────
if img_rgb is not None and not run_btn and not demo_choice:
    st.image(to_pil(img_rgb), caption=img_label, width=400)
    st.caption("Click **▶ Run Assessment** to analyse this image.")

# ── RUN INFERENCE ─────────────────────────────────────────────────────────
if (run_btn and img_rgb is not None) or (demo_choice and img_rgb is not None):

    with st.status("Running pipeline...", expanded=True) as status:
        st.write("🔧 Enhancing image...")
        try:
            # Save to temp file (inference.py takes a file path)
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                tmp_path = tmp.name
                cv2.imwrite(tmp_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

            st.write("🔬 Running U-Net segmentation...")
            st.write("🏷️ Classifying damage type...")
            st.write("🌡️ Generating Grad-CAM heatmap...")

            images, result = inference.run_assessment(
                tmp_path, apply_enhancement=apply_enhancement
            )
            os.unlink(tmp_path)

            st.write("✅ Pipeline complete!")
            status.update(label="Assessment complete!", state="complete")

        except Exception as e:
            status.update(label="Assessment failed", state="error")
            st.error(f"Inference error: {e}")
            st.stop()

    # ── RESULTS ───────────────────────────────────────────────────────────
    damage_class = result["damage_class"]
    confidence   = result["confidence"]
    severity     = result["severity"]
    damage_pct   = result["damage_percentage"]

    # Metric cards
    st.markdown(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="metric-label">Damage Type</div>
            <div class="metric-value amber">{damage_class}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Confidence</div>
            <div class="metric-value {'green' if confidence > 0.7 else 'amber'}">{confidence:.1%}</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Severity</div>
            <div class="metric-value">
                {severity_badge(severity)}
            </div>
        </div>
        <div class="metric-card">
            <div class="metric-label">Area Damaged</div>
            <div class="metric-value {'red' if damage_pct > 20 else 'amber'}">{damage_pct:.1f}%</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Pipeline image strip
    st.markdown("#### Pipeline output")
    c1, c2, c3, c4, c5 = st.columns(5)
    strip = [
        (c1, "original",  "Original"),
        (c2, "enhanced",  "Enhanced"),
        (c3, "mask",      "Damage Mask"),
        (c4, "overlay",   "Overlay"),
        (c5, "gradcam",   "Grad-CAM"),
    ]
    for col, key, label in strip:
        arr = images.get(key)
        if arr is not None:
            col.image(to_pil(arr), use_container_width=True)
            col.markdown(f'<div class="step-label">{label}</div>', unsafe_allow_html=True)

    # Recommendations
    st.markdown("#### Repair recommendations")
    for rec in result.get("recommendations", []):
        st.markdown(f'<div class="rec-item">→ {rec}</div>', unsafe_allow_html=True)

    # Class probabilities
    st.markdown("#### All class probabilities")
    probs = result.get("all_probs", {})
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    prob_html = ""
    for cls_name, prob in sorted_probs:
        pct = prob * 100
        prob_html += f"""
        <div class="prob-row">
            <span class="prob-name">{cls_name}</span>
            <div class="prob-bar"><div class="prob-fill" style="width:{pct:.1f}%"></div></div>
            <span class="prob-pct">{pct:.1f}%</span>
        </div>"""
    st.markdown(prob_html, unsafe_allow_html=True)

    # PDF download
    st.markdown("#### Download report")
    pdf_bytes = generate_pdf(result, images)
    st.download_button(
        label="⬇ Download PDF Report",
        data=pdf_bytes,
        file_name=f"damage_report_{damage_class.lower().replace(' ', '_')}.pdf",
        mime="application/pdf",
        use_container_width=True,
    )

elif run_btn and img_rgb is None:
    st.warning("Please upload an image or select a demo before running.")

# ── Footer ─────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    '<p style="text-align:center;color:#334155;font-size:0.75rem;'
    'font-family:JetBrains Mono,monospace;">'
    'U-Net segmentation · EfficientNetB0 · Grad-CAM · CarDD dataset · '
    'Built by <a href="https://github.com/duttapranjal" style="color:#475569;">duttapranjal</a>'
    '</p>',
    unsafe_allow_html=True
)
