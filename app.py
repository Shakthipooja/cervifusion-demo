
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from pathlib import Path

st.set_page_config(page_title="CerviFusion", page_icon="🔬", layout="centered")

MODEL_PATH = Path("cervifusion_model.keras")
CLASS_PATH = Path("class_names.json")
IMG_SIZE = (224, 224)

st.title("CerviFusion")
st.caption("Educational cervical-cell image classification prototype")
st.warning(
    "Research/educational use only. This system is not a medical device "
    "and must not be used to diagnose, screen, or make treatment decisions."
)

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH, "r") as f:
        class_names = json.load(f)
    return model, class_names

try:
    model, class_names = load_artifacts()
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

uploaded = st.file_uploader(
    "Upload a cell image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

if uploaded is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded image", use_container_width=True)

        # Basic file/image quality check only; not true out-of-distribution detection.
        if image.width < 32 or image.height < 32:
            st.error("Image is too small to analyze.")
            st.stop()

        resized = image.resize(IMG_SIZE)
        arr = np.asarray(resized, dtype=np.float32)
        arr = np.expand_dims(arr, axis=0)

        probabilities = model.predict(arr, verbose=0)[0]
        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])

        st.subheader("Model output")
        st.write("Predicted dataset class:", class_names[best_index])
        st.write(f"Model confidence: {confidence:.1%}")

        st.caption(
            "Confidence is not a probability of disease and does not establish "
            "clinical reliability."
        )

        st.subheader("Class scores")
        for name, score in zip(class_names, probabilities):
            st.write(f"{name}: {score:.1%}")

    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file could not be read as an image.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
