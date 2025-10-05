import io, os, json, tempfile, requests
import numpy as np
import pandas as pd
from PIL import Image
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
AUTO_PRECISION_TARGET = 0.90               # the "promise" for automation

st.set_page_config(page_title="Infractions HITL", page_icon="📡", layout="wide")

# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data
def load_labels_and_thresholds():
    labels = pd.read_csv(f"{ART_DIR}/label_names_v3_2.csv", header=None)[0].astype(str).tolist()
    thr_deploy = np.loadtxt(f"{ART_DIR}/best_thresholds_tf_v3_2.csv", delimiter=",", dtype=float)
    # Automation thresholds are optional
    auto_thr_path = f"{ART_DIR}/automation_thr_hi_v3_2.csv"
    auto_thr = np.loadtxt(auto_thr_path, delimiter=",", dtype=float) if os.path.exists(auto_thr_path) else None
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
def preprocess_image(img_pil, size=IMG_SIZE):
    # If your saved model includes preprocessing layers, raw 0..255 RGB is fine.
    # Otherwise, normalize to [0,1] here.
    img = img_pil.convert("RGB").resize(size)
    arr = np.asarray(img).astype("float32") / 255.0
    return arr

def predict_image(model, img_pil):
    arr = preprocess_image(img_pil)  # HxWxC in [0,1]
    x_img = np.expand_dims(arr, axis=0)  # [1,H,W,3]
    # Tabular tower: if your model expects tabular too, pass zeros or supply real features here.
    # We assume the saved model encapsulates fusion inputs or only image; adjust if needed:
    try:
        prob = model.predict(x_img, verbose=0)
    except Exception as e:
        # Backward compatibility: some fusion models expect [image, tabular]
        x_tab = np.zeros((1, 361), dtype="float32")
        prob = model.predict([x_img, x_tab], verbose=0)
    prob = np.asarray(prob).reshape(1, -1)  # [1, L]
    return prob[0]  # [L]

# -----------------------------
# Decision policy (HITL)
# -----------------------------
def apply_thresholds(probs, thr_vec):
    return (probs >= thr_vec).astype(int)

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
    L = len(labels)
    # Deploy thresholds (VAL-tuned) for general classification
    hit = apply_thresholds(probs, thr_vec).astype(bool)

    # Enforce per-sample Top-K after thresholding to cap sprawl
    k_mask = topk_mask(probs, topk)
    hit = hit & k_mask

    df = pd.DataFrame({
        "label": labels,
        "prob": probs,
        "thr_deploy": thr_vec,
        "over_thr": hit
    }).sort_values("prob", ascending=False)

    # Automation branch (only if hi-precision thresholds exist)
    auto_decision = "REVIEW"
    auto_used = False
    auto_labels = []

    if auto_thr is not None:
        # Allow only predictions above (auto_thr + margin) and within Top-K
        auto_hit = (probs >= (auto_thr + margin)) & k_mask
        if auto_hit.any():
            auto_decision = "AUTO_FLAG"
            auto_labels = [labels[i] for i in np.where(auto_hit)[0]]
        else:
            # (Optional) AUTO_CLEAR if all probs are very low; keep conservative default off
            auto_decision = "REVIEW"
        auto_used = True

    # If no automation thresholds available or feasible, stay REVIEW
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
    topk = st.number_input("Top-K per image", min_value=1, max_value=min(5, len(labels)), value=TOPK_DEFAULT, step=1)
    margin = st.slider("Automation abstain margin", 0.0, 0.1, ABSTAIN_DEFAULT, 0.01)
    st.write(f"**Deploy thresholds:** `{len(thr_deploy)}` labels")
    st.write("**Automation:**", "available" if auto_thr is not None else "disabled (no feasible ≥90% policy)")

tab1, tab2 = st.tabs(["Single Image", "Batch"])

# --- Single image tab ---
with tab1:
    uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png","bmp","webp"])
    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded", use_column_width=True)
        with st.spinner("Predicting..."):
            probs = predict_image(model, img)
            df, decision, ctx = decision_for_sample(probs, labels, thr_deploy, topk=topk, margin=margin, auto_thr=auto_thr)

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
                df, decision, ctx = decision_for_sample(probs, labels, thr_deploy, topk=topk, margin=margin, auto_thr=auto_thr)
                top = df.head(topk)[["label","prob"]].to_dict("records")
                rows.append({
                    "file": f.name,
                    "decision": decision,
                    "top_labels": "; ".join([f"{r['label']} ({r['prob']:.2f})" for r in top])
                })
            except Exception as e:
                rows.append({"file": f.name, "decision": f"ERROR: {e}", "top_labels": ""})
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        st.download_button("Download decisions (CSV)",
                           data=pd.DataFrame(rows).to_csv(index=False),
                           file_name="decisions.csv",
                           mime="text/csv")
