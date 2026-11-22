"""
app.py
------
Streamlit UI for Pneumonia Detection System.
Upload a chest X-ray → get prediction, triage level, and Grad-CAM heatmap.

Run:
    streamlit run app.py
"""

import os
import sys
import numpy as np
import cv2
from PIL import Image
import torch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import streamlit as st

# ── Page config ─────────────────────────────
st.set_page_config(
    page_title="Pneumonia Detection System",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, os.path.dirname(__file__))
import config
from model import get_model
from utils import load_checkpoint
from dataset import get_transforms
from triage import assign_triage_level

# ── Custom CSS ──────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .block-container { padding-top: 1.5rem; }

    .title-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border-left: 5px solid #58a6ff;
        padding: 20px 28px;
        border-radius: 10px;
        margin-bottom: 24px;
    }
    .title-box h1 { color: #58a6ff; margin: 0; font-size: 2rem; }
    .title-box p  { color: #8b949e; margin: 6px 0 0 0; font-size: 1rem; }

    .metric-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
    }
    .metric-card .val { font-size: 2.2rem; font-weight: bold; }
    .metric-card .lbl { color: #8b949e; font-size: 0.85rem; margin-top: 4px; }

    .triage-high {
        background: linear-gradient(135deg, #2d1b1b, #3d1f1f);
        border: 2px solid #f85149;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .triage-uncertain {
        background: linear-gradient(135deg, #2d2a1b, #3d361f);
        border: 2px solid #d29922;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .triage-low {
        background: linear-gradient(135deg, #1b2d1b, #1f3d1f);
        border: 2px solid #3fb950;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .triage-title { font-size: 1.8rem; font-weight: bold; margin-bottom: 8px; }
    .triage-desc  { font-size: 0.95rem; color: #c9d1d9; margin-bottom: 10px; }
    .triage-action { font-size: 1rem; font-weight: bold; }

    .info-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 18px;
        margin-top: 10px;
        font-size: 0.9rem;
        color: #c9d1d9;
    }
    .prob-bar-bg {
        background: #21262d;
        border-radius: 20px;
        height: 22px;
        margin: 6px 0;
        overflow: hidden;
    }
    .section-header {
        color: #58a6ff;
        font-size: 1.1rem;
        font-weight: bold;
        border-bottom: 1px solid #30363d;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
</style>
""", unsafe_allow_html=True)


# ── Load model (cached) ─────────────────────
@st.cache_resource
def load_model(model_name):
    config.MODEL_NAME = model_name
    ckpt_path = os.path.join(config.OUTPUT_DIR, f"best_model_{model_name}.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = config.MODEL_PATH
    model = get_model(name=model_name)
    model, _, _, _ = load_checkpoint(ckpt_path, model)
    model.eval()
    return model


# ── Grad-CAM ────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach())
        )
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradients', go[0].detach())
        )

    def generate(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = output.argmax(dim=1).item()
        pred_prob = probs[0, pred_idx].item()
        self.model.zero_grad()
        output[0, pred_idx].backward()
        pooled = self.gradients.mean(dim=[2, 3], keepdim=True)
        weighted = (pooled * self.activations).squeeze(0).sum(0).cpu().numpy()
        heatmap = np.maximum(weighted, 0)
        if heatmap.max() > 0:
            heatmap /= heatmap.max()
        return heatmap, pred_idx, pred_prob


def get_target_layer(model, model_name):
    name = model_name.lower()
    if "densenet" in name:
        return model.features.denseblock4
    elif "resnet" in name:
        return model.layer4
    elif "efficientnet" in name:
        return model.features[-1]
    elif "vgg" in name:
        return model.features[-1]
    elif "mobilenet" in name:
        return model.features[-1]


def overlay_heatmap(heatmap, orig_img):
    h, w = orig_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    colored = (cm.get_cmap("jet")(heatmap_resized)[:, :, :3] * 255).astype(np.uint8)
    blended = cv2.addWeighted(orig_img, 0.6, colored, 0.4, 0)
    return blended, heatmap_resized


# ── Predict ─────────────────────────────────
@torch.no_grad()
def predict(image_pil, model, model_name, high_t, low_t):
    transform = get_transforms("test")
    input_tensor = transform(image_pil).unsqueeze(0).to(config.DEVICE)

    # Grad-CAM
    target_layer = get_target_layer(model, model_name)
    grad_cam = GradCAM(model, target_layer)

    with torch.enable_grad():
        heatmap, pred_idx, pred_prob_target = grad_cam.generate(input_tensor)

    # Full probs
    output = model(input_tensor)
    probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
    pneumonia_prob = float(probs[1])
    normal_prob = float(probs[0])

    triage = assign_triage_level(pneumonia_prob, high_t, low_t)

    # Overlay
    orig_np = np.array(image_pil.resize((config.IMAGE_SIZE, config.IMAGE_SIZE)))
    blended, heatmap_resized = overlay_heatmap(heatmap, orig_np)

    return {
        "pneumonia_prob": pneumonia_prob * 100,
        "normal_prob": normal_prob * 100,
        "pred_class": config.CLASSES[pred_idx],
        "triage": triage,
        "orig_np": orig_np,
        "heatmap": heatmap_resized,
        "blended": blended,
    }


# ── Sidebar ─────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    st.markdown("---")

    available_models = []
    for m in ["densenet121", "resnet50", "efficientnet_b0", "vgg16", "mobilenet_v3"]:
        p1 = os.path.join(config.OUTPUT_DIR, f"best_model_{m}.pth")
        p2 = config.MODEL_PATH
        if os.path.exists(p1) or (m == "densenet121" and os.path.exists(p2)):
            available_models.append(m)

    model_name = st.selectbox(
        "🧠 Select Model",
        available_models,
        index=0,
        help="Choose which trained model to use for prediction"
    )

    st.markdown("---")
    st.markdown("### 🎚️ Triage Thresholds")
    high_t = st.slider("HIGH RISK threshold", 0.5, 0.99, 0.80, 0.01,
                       help="P(Pneumonia) above this → HIGH RISK")
    low_t = st.slider("LOW RISK threshold", 0.01, 0.49, 0.20, 0.01,
                      help="P(Pneumonia) below this → LOW RISK")

    st.markdown("---")
    st.markdown("### 📊 Threshold Guide")
    st.markdown(f"""
    | Level | Condition |
    |---|---|
    | 🔴 HIGH RISK | P ≥ {high_t:.0%} |
    | 🟡 UNCERTAIN | {low_t:.0%} < P < {high_t:.0%} |
    | 🟢 LOW RISK | P ≤ {low_t:.0%} |
    """)

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Pneumonia Detection System**

    Built by:
    - Katukuri Sri Saitejaa Reddy
    - Vasamshetti Jayaganesh

    Amrita Vishwa Vidyapeetham
    """)


# ── Main Page ───────────────────────────────
st.markdown("""
<div class="title-box">
  <h1>🫁 Pneumonia Detection System</h1>
  <p>Upload a chest X-ray image to get an AI-assisted diagnosis with confidence-calibrated triage</p>
</div>
""", unsafe_allow_html=True)

# Upload
uploaded_file = st.file_uploader(
    "📂 Upload Chest X-Ray Image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPEG or PNG chest X-ray image"
)

if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")

    with st.spinner(f"🔄 Analysing with {model_name}..."):
        model = load_model(model_name)
        result = predict(image_pil, model, model_name, high_t, low_t)

    triage = result["triage"]
    level  = triage["level_key"]

    # ── Triage Banner ────────────────────────
    css_class = {
        "HIGH_RISK": "triage-high",
        "UNCERTAIN": "triage-uncertain",
        "LOW_RISK":  "triage-low",
    }[level]

    color = {
        "HIGH_RISK": "#f85149",
        "UNCERTAIN": "#d29922",
        "LOW_RISK":  "#3fb950",
    }[level]

    st.markdown(f"""
    <div class="{css_class}">
      <div class="triage-title" style="color:{color}">{triage['label']}</div>
      <div class="triage-desc">{triage['description']}</div>
      <div class="triage-action" style="color:{color}">⚡ {triage['action']}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Metrics Row ──────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
          <div class="val" style="color:{color}">{result['pneumonia_prob']:.1f}%</div>
          <div class="lbl">P(Pneumonia)</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
          <div class="val" style="color:#3fb950">{result['normal_prob']:.1f}%</div>
          <div class="lbl">P(Normal)</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
          <div class="val" style="color:#58a6ff">{result['pred_class']}</div>
          <div class="lbl">Prediction</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
          <div class="val" style="color:#bc8cff">{model_name}</div>
          <div class="lbl">Model Used</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Probability Bars ─────────────────────
    st.markdown('<div class="section-header">📊 Confidence Breakdown</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**PNEUMONIA**")
        st.progress(result['pneumonia_prob'] / 100)
        st.markdown(f"<p style='color:#f85149;font-size:1.2rem;font-weight:bold'>{result['pneumonia_prob']:.2f}%</p>", unsafe_allow_html=True)
    with col_b:
        st.markdown("**NORMAL**")
        st.progress(result['normal_prob'] / 100)
        st.markdown(f"<p style='color:#3fb950;font-size:1.2rem;font-weight:bold'>{result['normal_prob']:.2f}%</p>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Images Row ───────────────────────────
    st.markdown('<div class="section-header">🔬 Visual Analysis</div>', unsafe_allow_html=True)
    img_c1, img_c2, img_c3 = st.columns(3)

    with img_c1:
        st.markdown("**Original X-Ray**")
        st.image(result['orig_np'], use_container_width=True)

    with img_c2:
        st.markdown("**Grad-CAM Heatmap**")
        fig, ax = plt.subplots(figsize=(4, 4), facecolor='#0e1117')
        ax.imshow(result['heatmap'], cmap='jet')
        ax.axis('off')
        st.pyplot(fig, use_container_width=True)
        plt.close()

    with img_c3:
        st.markdown("**Overlay**")
        st.image(result['blended'], use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Clinical Info Box ────────────────────
    st.markdown('<div class="section-header">🏥 Clinical Guidance</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="info-box">
    <b>Triage Level:</b> {triage['label']}<br>
    <b>Pneumonia Probability:</b> {result['pneumonia_prob']:.2f}%<br>
    <b>Normal Probability:</b> {result['normal_prob']:.2f}%<br>
    <b>Recommended Action:</b> {triage['action']}<br><br>
    ⚠️ <i>This system is an AI-assisted decision support tool only.
    All predictions must be verified by a qualified clinician before clinical action.</i>
    </div>
    """, unsafe_allow_html=True)

else:
    # ── Welcome Screen ───────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="metric-card">
          <div class="val" style="color:#3fb950">🟢</div>
          <div style="font-size:1.1rem;font-weight:bold;margin:8px 0">LOW RISK</div>
          <div class="lbl">P(Pneumonia) ≤ 20%<br>Safe to deprioritise</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="metric-card">
          <div class="val" style="color:#d29922">🟡</div>
          <div style="font-size:1.1rem;font-weight:bold;margin:8px 0">UNCERTAIN</div>
          <div class="lbl">20% < P < 80%<br>Refer to radiologist</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="metric-card">
          <div class="val" style="color:#f85149">🔴</div>
          <div style="font-size:1.1rem;font-weight:bold;margin:8px 0">HIGH RISK</div>
          <div class="lbl">P(Pneumonia) ≥ 80%<br>Urgent treatment required</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👆 Upload a chest X-ray image using the file uploader above to get started.")

    st.markdown("### 🧠 Available Models")
    model_info = {
        "densenet121":     ("DenseNet121",      "7M",   "94.39%", "98.29%", "⭐ Best"),
        "resnet50":        ("ResNet50",          "25M",  "~92%",   "~97%",   ""),
        "efficientnet_b0": ("EfficientNet-B0",   "5M",   "~93%",   "~97.5%", ""),
        "vgg16":           ("VGG16",             "138M", "~90%",   "~95%",   ""),
        "mobilenet_v3":    ("MobileNetV3-Large", "5M",   "~91%",   "~96%",   "Fastest"),
    }
    rows = []
    for key, (name, params, acc, auc, note) in model_info.items():
        exists = os.path.exists(os.path.join(config.OUTPUT_DIR, f"best_model_{key}.pth"))
        status = "✅ Ready" if exists else "⏳ Not trained"
        rows.append({"Model": name, "Params": params, "Accuracy": acc, "AUC-ROC": auc, "Note": note, "Status": status})

    import pandas as pd
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
