import io, os, json, tempfile, requests
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import streamlit as st
import tensorflow as tf

# -----------------------------
# Configuration
# -----------------------------
ART_DIR      = os.environ.get("ARTIFACT_DIR", "artifacts")
MODEL_PATH   = os.environ.get("MODEL_PATH", "models/Pole_Infraction_Model5.keras")
MODEL_URL    = os.environ.get("MODEL_URL")  # optional: direct link to .keras (GitHub Release, S3, etc.)
IMG_SIZE     = (224, 224)                   # match training
TOPK_DEFAULT = 3
ABSTAIN_DEFAULT = 0.02
AUTO_PRECISION_TARGET = 0.90                # the "promise" for automation
TAB_DIM = int(os.environ.get("TAB_DIM", "361"))  # fallback tabular width if the model needs it

st.set_page_config(page_title="Infractions HITL", page_icon="📡", layout="wide")

# -----------------------------
# Helpers
# -----------------------------
def _flatten_1d(x) -> np.ndarray:
    """Flatten csv/np.loadtxt outputs to shape (L,)"""
    x = np.asarray(x)
    if x.ndim == 2 and 1 in x.shape:
        x = x.reshape(-1)
    elif x.ndim > 1:
        x = x.squeeze()
    return x.astype(np.float32)

# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data
def load_labels_and_thresholds():
    labels_path = f"{ART_DIR}/label_names_v3_2.csv"
    thr_path    = f"{ART_DIR}/best_thresholds_tf_v3_2.csv"

    if not os.path.exists(labels_path) or not os.path.exists(thr_path):
        st.error(
            "Missing required artifacts.\n\n"
            f"Expected:\n• {labels_path}\n• {thr_path}\n\n"
            "Please add them to the repo or set ARTIFACT_DIR."
        )
        st.stop()

    labels = pd.read_csv(labels_path, header=None)[0].astype(str).tolist()
    thr_deploy = _flatten_1d(np.loadtxt(thr_path, delimiter=","))

    if len(thr_deploy) != len(labels):
        st.error(
            f"Threshold vector length ({len(thr_deploy)}) does not match label count ({len(labels)}). "
            "Fix artifacts or update app paths."
        )
        st.stop()

    # Automation thresholds are optional
    auto_thr_path = f"{ART_DIR}/automation_thr_hi_v3_2.csv"
    auto_thr = None
    if os.path.exists(auto_thr_path):
        cand = _flatten_1d(np.loadtxt(auto_thr_path, delimiter=","))
        if len(cand) == len(labels):
            auto_thr = cand
        else:
            st.warning(
                "automation_thr_hi_v3_2.csv found but length does not match labels; "
                "automation will be disabled."
            )
            auto_thr = None

    return labels, thr_deploy, auto_thr

@st.cache_resource(show_spinner=True)
def load_model():
    path = MODEL_PATH
    if (not os.path.exists(path)) and MODEL_URL:
        # download once per cold start
        resp = requests.get(MODEL_URL, timeout=60)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(resp.content)
    model = tf.keras.models.load_model(path)
    return model

# -----------------------------
# Preprocess & inference
# -----------------------------
def preprocess_image(img_pil: Image.Image, size=IMG_SIZE) -> np.ndarray:
    """
    - Honor EXIF orientation
    - Force RGB (3 channels)
    - Exact 224x224
    - Normalize to [0,1]
    """
    img = ImageOps.exif_transpose(img_pil)
    img = img.convert("RGB").resize(size, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # (224,224,3)
    # Safety checks (if anything sneaks through as single-channel)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    elif arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return arr  # HWC

def predict_image(model, img_pil: Image.Image) -> np.ndarray:
    arr = preprocess_image(img_pil)                # (224,224,3)
    x_img = np.expand_dims(arr, axis=0)            # [1,H,W,3]
    # Try single-input first
    try:
        out = model(x_img, training=False)
        prob = np.asarray(out)
        if prob.ndim == 1:
            prob = prob[None, ...]
        elif prob.ndim > 2:
            prob = prob.reshape((prob.shape[0], -1))
        return prob[0]
    except Exception:
        # Fallback: models that expect [image, tabular]
        x_tab = np.zeros((1, TAB_DIM), dtype=np.float32)
        out = model([x_img, x_tab], training=False)
        prob = np.asarray(out)
        if prob.ndim == 1:
            prob = prob[None, ...]
        elif prob.ndim > 2:
            prob = prob.reshape((prob.shape[0], -1))
        return prob[0]

# -----------------------------
# Decision policy (HITL)
# -----------------------------
def apply_thresholds(probs, thr_vec):
    probs = np.asarray(probs, dtype=np.float32)
    thr_vec = np.asarray(thr_vec, dtype=np.float32)
    return (probs >= thr_vec).astype(bool)

def topk_mask(probs, k):
    idx = np.argsort(-probs)[:k]
    m = np.zeros_like(probs, dtype=bool)
    m[idx] = True
    return m

def decision_for_sample(probs, labels, thr_vec, topk=3, margin=0.02, auto_thr=None):
    """
    Returns:
      df (per-label view),
      decision (AUTO_FLAG / REVIEW / AUTO_CLEAR),
      auto_context (which thresholding was used)
    """
    probs = np.asarray(probs, dtype=np.float32)
    L = len(labels)
    assert probs.shape[0] == L, f"probs length {probs.shape[0]} != labels {L}"

    # Deploy thresholds (VAL-tuned)
    hit = apply_thresholds(probs, thr_vec)

    # Enforce per-sample Top-K to cap sprawl
    k_mask = topk_mask(probs, topk)
    hit = hit & k_mask

    df = pd.DataFrame({
        "label": labels,
        "prob": probs,
        "thr_deploy": np.asarray(thr_vec, dtype=np.float32),
        "over_thr": hit
    }).sort_values("prob", ascending=False)

    # Automation branch (only if hi-precision thresholds exist)
    auto_decision = "REVIEW"
    auto_used = False
    auto_labels = []

    if auto_thr is not None:
        auto_thr = np.asarray(auto_thr, dtype=np.float32)
        auto_hit = (probs >= (auto_thr + margin)) & k_mask
        if auto_hit.any():
            auto_decision = "AUTO_FLAG"
            auto_labels = [labels[i] for i in np.where(auto_hit)[0]]
        auto_used = True

    return df, (auto_decision if auto_used else "REVIEW"), {"auto_thr_used": auto_used, "auto_labels": auto_labels}

# -----------------------------
# UI
# -----------------------------
st.title("📡 Utility Pole Infractions — HITL Prediction App")
st.caption("Keras model + deploy thresholds (VAL-tuned). Automation honors a ≥90% precision promise when feasible; otherwise all predictions go to REVIEW.")

with st.sidebar:
    st.header("Settings")
    labels, thr_deploy, auto_thr = load_labels_and_thresholds()
    model = load_model()

    max_topk = min(5, len(labels))
    topk = st.number_input("Top-K per image", min_value=1, max_value=max_topk, value=min(TOPK_DEFAULT, max_topk), step=1)
    margin = st.slider("Automation abstain margin", 0.0, 0.1, ABSTAIN_DEFAULT, 0.01)
    st.write(f"**Deploy thresholds:** `{len(thr_deploy)}` labels")
    st.write("**Automation:**", "available" if auto_thr is not None else "disabled (no feasible ≥90% policy)")

tab1, tab2 = st.tabs(["Single Image", "Batch"])

# --- Single image tab ---
with tab1:
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(ImageOps.exif_transpose(img), caption="Uploaded", use_column_width=True)
        with st.spinner("Predicting..."):
            probs = predict_image(model, img)
            df, decision, ctx = decision_for_sample(
                probs, labels, thr_deploy, topk=topk, margin=margin, auto_thr=auto_thr
            )

        colA, colB = st.columns([2,1])
        with colA:
            st.subheader("Per-label predictions")
            st.dataframe(df.style.format({"prob":"{:.3f}","thr_deploy":"{:.3f}"}), use_container_width=True)
        with colB:
            st.subheader("Decision")
            if ctx["auto_thr_used"]:
                if decision == "AUTO_FLAG":
                    st.success("AUTO_FLAG ✅ (meets ≥90% precision policy)", icon="✅")
                    st.write("Labels:", ", ".join(ctx["auto_labels"]) or "—")
                else:
                    st.warning("REVIEW (automation abstained)", icon="🛈")
                    st.write("No labels cleared the hi-precision bar.")
            else:
                st.info("REVIEW (no feasible hi-precision policy available)", icon="👀")

# --- Batch tab (multi-image) ---
with tab2:
    multi = st.file_uploader("Upload multiple images", type=["jpg","jpeg","png","bmp","webp"], accept_multiple_files=True)
    if multi:
        rows = []
        for f in multi:
            try:
                img = Image.open(f)
                probs = predict_image(model, img)
                df, decision, ctx = decision_for_sample(
                    probs, labels, thr_deploy, topk=topk, margin=margin, auto_thr=auto_thr
                )
                top = df.head(topk)[["label","prob"]].to_dict("records")
                rows.append({
                    "file": getattr(f, "name", "image"),
                    "decision": decision,
                    "top_labels": "; ".join([f"{r['label']} ({r['prob']:.2f})" for r in top])
                })
            except Exception as e:
                rows.append({"file": getattr(f, "name", "image"), "decision": f"ERROR: {e}", "top_labels": ""})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.download_button("Download decisions (CSV)",
                           data=pd.DataFrame(rows).to_csv(index=False),
                           file_name="decisions.csv",
                           mime="text/csv")
