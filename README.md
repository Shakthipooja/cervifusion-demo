# cervifusion-demo
An educational deep-learning prototype for classifying cervical-cell images into five dataset categories, with a Streamlit web interface for image upload and model predictions.
# CerviFusion

### Cervical-Cell Image Classification Research Prototype

CerviFusion is an educational deep-learning project that classifies cervical-cell
images into five dataset categories using a trained image-classification model.
The project includes a Streamlit web application where users can upload an image
and view the model's predicted class and class scores.

> **Disclaimer:** This project is for educational and research purposes only.
> It is not a medical device and must not be used for diagnosis, screening,
> treatment decisions, or other clinical use. Model predictions are not a
> substitute for evaluation by a qualified healthcare professional.

## Features

- Upload a supported image through a web interface.
- Predict one of the five dataset categories.
- Display the model's class scores.
- Show the uploaded image alongside the prediction.
- Provide a simple interface built with Streamlit.

## Dataset categories

The model is trained to classify images into these dataset labels:

1. `im_Dyskeratotic`
2. `im_Koilocytotic`
3. `im_Metaplastic`
4. `im_Parabasal`
5. `im_Superficial-Intermediate`

These are dataset categories, not cancer diagnoses.

## Technology

- Python
- TensorFlow / Keras
- Streamlit
- NumPy
- Pillow

## Repository files

- `app.py` — Streamlit application.
- `requirements.txt` — Python dependencies.
- `class_names.json` — Class labels in the order used by the model.
- `cervifusion_model.keras` — Trained Keras model.

## Run locally

### 1. Clone the repository


streamlit
tensorflow
numpy
pillow
huggingface_hub

