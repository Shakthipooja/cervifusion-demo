import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
from huggingface_hub import hf_hub_download

st.set_page_config(page_title="CerviFusion", page_icon="🔬")

st.title("CerviFusion")
st.caption("Cervical-cell image classification research prototype")

st.warning(
    "Educational/research use only. This is not a medical device and "
    "must not be used for diagnosis, screening, or treatment decisions."
)

# Hugging Face repository and exact model filename
MODEL_REPO = "Shakthipooja/cervifusion-model"
MODEL_FILENAME = "cervifusion_model (1).keras"

# This is only a rough confidence filter, not an invalid-image detector.
CONFIDENCE_THRESHOLD = 0.70


@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILENAME,
    )
    return tf.keras.models.load_model(model_path)


@st.cache_data
def load_class_names():
    with open("class_names.json", "r", encoding="utf-8") as file:
        return json.load(file)


try:
    model = load_model()
    class_names = load_class_names()

    if len(class_names) != model.output_shape[-1]:
        st.error(
            "The number of class names does not match the model output. "
            "Check class_names.json and the trained model."
        )
        st.stop()

except Exception as error:
    st.error(f"Could not load model files: {error}")
    st.stop()


uploaded_file = st.file_uploader(
    "Upload a cervical-cell image",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        if image.width < 32 or image.height < 32:
            st.error("Image is too small to process.")
            st.stop()

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True,
        )

        # This preprocessing must match the preprocessing used during training.
        resized_image = image.resize((224, 224))
        image_array = np.asarray(resized_image, dtype=np.float32)[None, ...]

        scores = model.predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(scores))
        top_score = float(scores[predicted_index])

        st.subheader("Model output")

        if top_score < CONFIDENCE_THRESHOLD:
            st.error(
                "Uncertain result. The image may not match the supported "
                "dataset classes. Please upload a cervical-cell image."
            )
        else:
            st.write("Predicted dataset class:", class_names[predicted_index])
            st.write(f"Model score: {top_score:.1%}")

        st.subheader("Scores by class")
        for name, score in zip(class_names, scores):
            st.write(f"{name}: {score:.1%}")

        st.caption(
            "The model score is not a probability of disease or a clinical diagnosis. "
            "A high score does not guarantee that the uploaded image is a cervical-cell image."
        )

    except (UnidentifiedImageError, OSError):
        st.error("The uploaded file could not be read as an image.")
    except Exception as error:
        st.error(f"Prediction failed: {error}")
        
