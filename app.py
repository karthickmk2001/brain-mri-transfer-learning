import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)
CLASSES = ["no_tumor", "tumor"]

MODELS = {
    "Feature Extraction (68.63% test accuracy, recommended)": {
        "path": "feature_extraction_model.keras",
        "accuracy": 68.63,
        "loss": 0.6055,
        "note": "All ResNet50 layers frozen, only the classifier head was trained. "
                "Best overall accuracy and safer tumor recall on the test set.",
    },
    "Fine-Tuned (62.75% test accuracy)": {
        "path": "finetuned_model.keras",
        "accuracy": 62.75,
        "loss": 0.6892,
        "note": "Top 30 ResNet50 layers were unfrozen and retrained at a low learning rate. "
                "On this small dataset it overfit and scored lower, missing more actual "
                "tumor cases (45% tumor recall vs 90% no_tumor recall) - shown here for comparison.",
    },
}

st.set_page_config(page_title="Brain MRI Tumor Classifier", page_icon="🧠", layout="centered")

st.title("🧠 Brain MRI Tumor Classifier")
st.caption(
    "Transfer learning (ResNet50) lab project - H9MLAI Machine Learning Lab · "
    "Karthick Subramanian Muthukkaruppan"
)
st.warning(
    "Educational lab project only, trained on a small public dataset "
    "(253 images). **Not a diagnostic tool - do not use for real medical decisions.**"
)


@st.cache_resource(show_spinner="Loading model...")
def load_model(path):
    return tf.keras.models.load_model(path)


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMG_SIZE)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


with st.sidebar:
    st.header("Model")
    choice = st.radio("Choose which trained model to use:", list(MODELS.keys()))
    info = MODELS[choice]
    st.metric("Test accuracy", f"{info['accuracy']}%")
    st.metric("Test loss", f"{info['loss']}")
    st.caption(info["note"])

model = load_model(info["path"])

uploaded = st.file_uploader("Upload a brain MRI scan (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded scan", use_container_width=True)

    x = preprocess(image)
    prob_tumor = float(model.predict(x, verbose=0)[0][0])
    predicted = CLASSES[int(prob_tumor > 0.5)]
    confidence = prob_tumor if predicted == "tumor" else 1 - prob_tumor

    with col2:
        if predicted == "tumor":
            st.error(f"### Prediction: Tumor\nConfidence: {confidence * 100:.1f}%")
        else:
            st.success(f"### Prediction: No Tumor\nConfidence: {confidence * 100:.1f}%")
        st.progress(confidence)
        st.caption(f"Raw model output (P(tumor)): {prob_tumor:.4f}")
else:
    st.info("Upload an image to get a prediction.")

st.divider()
st.caption(
    "Model: ResNet50 pretrained on ImageNet, fine-tuned/feature-extracted on the "
    "Kaggle 'Brain MRI Images for Brain Tumor Detection' dataset. "
    "See README.md and the lab notebook for full methodology and results."
)
