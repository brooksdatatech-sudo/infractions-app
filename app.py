import io, os, json, tempfile, requests
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
import keras

# -----------------------------
# Configuration
# -----------------------------
ART_DIR       = os.environ.get("ARTIFACT_DIR", "artifacts")
MODEL_PATH    = os.environ.get("MODEL_PATH", "models/Pole_Infraction_Model5.keras")
MODEL_URL     = os.environ.get("MODEL_URL")  # optional: direct link to .keras (Release/S3/etc.)
TOPK_DEFAULT  = 3
ABSTAIN_DEFAULT = 0.02
AUTO_PRECISION_TARGET = 0.90

st.set_page_config(page_title="Infractions HITL", page_icon="📡", layout="wide")

# -----------------------------
# Cached loaders
# -----------------------------
@st.cache_data
def load_labels_and_thresholds():
    labels = pd.read_csv(f"{ART_DIR}/label_names_v3_2.csv", header=None)[0].astype(str).tolist()
    thr_deploy = np.loadtxt(f"{ART_DIR}/best_thresholds_tf_v3_2.csv", delimiter=",", dtype=float)
    auto_thr_path = f"{ART_DIR}/automation_thr_hi_v3_2.csv"
    auto_thr = np.loadtxt(auto_thr_path, delimiter=",", dtype=float) if os.path.exists(auto_thr_path) else None
    return labels, thr_deploy, auto_thr

@st.cache_resource(show_spinner=True)
def load_model():
    path = MODEL_PATH
    if (not os.path.exists(path)) and MODEL_URL:
        resp = requests.get(MODEL_URL, timeout=60)
        resp.raise_for_status()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(resp.content)

    # Prefer Keras 3 native loader for .keras files
    try:
        model = keras.saving.load_model(path)  # Keras 3 native format
    except Exception as e_k3:
        # Fallback to tf.keras if the file happens to be a TF SavedModel/HDF5
        model = tf.keras.models.load_model(path)
    return model

# -----------------------------
# Model image spec helpers
# -----------------------------
def get_image_input_spec(model):
    """
    Returns (height, width, channels, input_index) for the first 4D input tensor.
    If model is fusion [image, tabular], picks the 4D one.
    """
    for i, t in enumerate(model.inputs):
        shape = t.shape
        if len(shape) == 4:
            h = int(shape[1]) if shape[1] is not None else 224
            w = int(shape[2]) if shape[2] is not None else 224
            c = int(shape[3]) if shape[3] is not None else 3
            return h, w, c, i
    # Fallback
    return 224, 224, 3, 0

def preprocess_image(img_pil, size_hw, channels):
    h, w = size_hw
    if channels == 1:
        img = img_pil.convert("L").resize((w, h))
        arr = np.asarray(img).astype("float32") / 255.0
        arr = np.expand_dims(arr, axis=-1)  # HWC with C=1
    else:
        img = img_pil.convert("RGB").resize((w, h))
        arr = np.asarray(img).astype("float32") / 255.0
    return arr

# -----------------------------
# Inference
# -----------------------------
def predict_image(model, img_pil):
    h, w, c, img_input_index = get_image_input_spec(model)
    arr = preprocess_image(img_pil, (h, w), c)
    x_img = np.expand_dims(arr, axis=0)  # [1,H,W,C]

    # Try single input
    try:
        prob = model.predict(x_img, verbose=0)
        prob = np.asarray(prob).reshape(1, -1)
        return prob[0]
    except Exception:
        # Fusion fallback: [image, tabular] any order
        # Build a zero tabular vector sized either from model input shape or default 361
        # Try both orders robustly.
        tab_dim = 361
        for inp in model.inputs:
            if len(inp.shape) == 2 and inp.shape[1] is not None:
                tab_dim = int(inp.shape[1])
                break
        x_tab = np.zeros((1, tab_dim), dtype="float32")

        # Try [image, tabular] then [tabular, image]
        orders = [
            ([x_img, x_tab],),
            ([x_tab, x_img],),
        ]
        last_err = None
        for (feed,) in orders:
            try:
                prob = model.predict(feed, verbose=0)
                prob = np.asarray(prob).reshape(1, -1)
                return prob[0]
            except Exception as e:
                last_err = e
        raise last_err

# -----------------------------
# Decision policy (HITL)
# -----------------------------
def apply_thresholds(probs, thr_vec): return (probs >= thr_vec).astype(int)

def topk_mask(probs, k):
    idx = np.argsort(-probs)[:k]
    m = np.zeros_like(probs, dtype=bool); m[idx] = True
    return m

def decision_for_sample(probs, labels, thr_vec, topk=3, margin=0.02, auto_thr=None):
    hit = apply_thresholds(probs, thr_vec).astype(bool)
    k_mask = topk_mask(probs, topk)
    hit = hit & k_mask

    df = pd.DataFrame({
        "label": labels,
        "prob": probs,
        "thr_deploy": thr_vec,
        "over_thr": hit
    }).sort_values("prob", ascending=False)

    auto_decision = "REVIEW"; auto_used = False; auto_labels = []
    if auto_thr is not None:
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
st.caption("Keras/TF model + VAL-tuned deploy thresholds. Automation only triggers when the ≥90% precision policy is feasible; otherwise it stays in REVIEW.")

with st.sidebar:
    st.header("Settings")
    labels, thr_deploy, auto_thr = load_labels_and_thresholds()
    model = load_model()
    # Show what the model expects
    mh, mw, mc, _ = get_image_input_spec(model)
    st.write(f"**Model image spec:** {mh}×{mw}×{mc}")
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
        out = pd.DataFrame(rows)
        st.dataframe(out, use_container_width=True)
        st.download_button("Download decisions (CSV)", data=out.to_csv(index=False), file_name="decisions.csv", mime="text/csv")
