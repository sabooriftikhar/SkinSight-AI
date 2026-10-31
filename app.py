import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf

from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

PROJECT_REPORT_URL = "https://github.com/sabooriftikhar/SkinSight-AI/blob/main/SkinSight_AI_Project_Report.docx"

PRESENTATION_URL = "https://github.com/sabooriftikhar/SkinSight-AI/blob/main/SkinSight_AI_Final_Presentation.pptx"

GITHUB_REPO_URL = "https://github.com/sabooriftikhar/SkinSight-AI"
# ----------------------------------
# PAGE CONFIGURATION
# ----------------------------------

st.set_page_config(
    page_title="SkinSight AI",
    page_icon="🔬",
    layout="wide"
)


# ----------------------------------
# PROJECT INFORMATION
# ----------------------------------

st.title("🔬 SkinSight AI")

st.subheader(
    "Deep Learning-Based Skin Lesion Image Classification"
)

st.write(
    "Upload a dermatoscopic skin image to see how the "
    "trained deep learning model classifies it."
)

st.warning(
    "Educational and research prototype only. "
    "This application is NOT a medical diagnostic tool. "
    "Predictions may be incorrect. Consult a qualified "
    "dermatologist for medical evaluation."
)


# ----------------------------------
# CLASS INFORMATION
# ----------------------------------

class_descriptions = {
    "akiec": "Actinic keratoses and intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions"
}


# ----------------------------------
# LOAD MODEL
# ----------------------------------

@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        "model.keras"
    )

    return model


@st.cache_data
def load_class_names():

    with open("class_names.json", "r") as f:
        return json.load(f)


@st.cache_data
def load_metrics():

    with open("metrics.json", "r") as f:
        return json.load(f)


try:
    model = load_model()
    class_names = load_class_names()
    metrics = load_metrics()

except Exception as e:

    st.error(
        f"Could not load project files: {e}"
    )

    st.stop()


# ----------------------------------
# SIDEBAR
# ----------------------------------

st.sidebar.title("About SkinSight AI")

st.sidebar.write(
    """
    SkinSight AI uses MobileNetV2 transfer learning
    to classify dermatoscopic images into seven
    predefined skin lesion categories.
    """
)

st.sidebar.markdown("---")

st.sidebar.write("**Model:** MobileNetV2")

st.sidebar.write("**Framework:** TensorFlow / Keras")

st.sidebar.write("**Image Size:** 160 × 160")

st.sidebar.write("**Classes:** 7")

st.sidebar.markdown("---")

st.sidebar.caption(
    "Academic Deep Learning Project"
)
st.sidebar.markdown("---")

st.sidebar.subheader("📚 Project Resources")

st.sidebar.markdown(
    f"""
    **📄 Project Report**

    [Open Word Report]({PROJECT_REPORT_URL})

    **📊 Project Presentation**

    [View PPT Slides]({PRESENTATION_URL})

    **💻 GitHub Repository**

    [View Source Code]({GITHUB_REPO_URL})
    """
)

st.sidebar.markdown("---")

st.sidebar.caption(
    "Developed as an Academic Deep Learning Project"
)

# ----------------------------------
# IMAGE UPLOAD
# ----------------------------------

st.header("Upload Skin Image")

uploaded_file = st.file_uploader(
    "Choose a JPG or PNG image",
    type=["jpg", "jpeg", "png"]
)


def prepare_image(image):

    image = image.convert("RGB")

    image = image.resize((160, 160))

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = preprocess_input(
        image_array
    )

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    return image_array


# ----------------------------------
# PREDICTION
# ----------------------------------

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:

        st.image(
            image,
            caption="Uploaded Skin Image",
            use_container_width=True
        )

    with col2:

        st.subheader("Classification Result")

        if st.button(
            "Analyze Image",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing image..."
            ):

                processed_image = prepare_image(
                    image
                )

                predictions = model.predict(
                    processed_image,
                    verbose=0
                )[0]

                predicted_index = int(
                    np.argmax(predictions)
                )

                predicted_class = class_names[
                    predicted_index
                ]

                confidence = float(
                    predictions[predicted_index]
                ) * 100

                st.success("Prediction completed!")

                st.metric(
                    "Predicted Class",
                    predicted_class.upper()
                )

                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )

                st.write(
                    class_descriptions.get(
                        predicted_class,
                        "Unknown category"
                    )
                )

                st.markdown("---")

                st.subheader("Top 3 Predictions")

                top_indices = np.argsort(
                    predictions
                )[-3:][::-1]

                for index in top_indices:

                    category = class_names[index]

                    probability = float(
                        predictions[index]
                    ) * 100

                    st.write(
                        f"**{category.upper()}** — "
                        f"{probability:.2f}%"
                    )

                    st.progress(
                        float(predictions[index])
                    )

                st.caption(
                    "These are model output probabilities, "
                    "not clinically validated probabilities "
                    "of having a disease."
                )


# ----------------------------------
# MODEL PERFORMANCE
# ----------------------------------

st.markdown("---")

st.header("Model Performance")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Validation Accuracy",
    f"{metrics['validation_accuracy'] * 100:.2f}%"
)

col2.metric(
    "Macro Precision",
    f"{metrics['precision_macro'] * 100:.2f}%"
)

col3.metric(
    "Macro F1 Score",
    f"{metrics['f1_macro'] * 100:.2f}%"
)


# ----------------------------------
# EVALUATION GRAPHS
# ----------------------------------

st.markdown("---")

st.header("Model Evaluation Visualizations")

graph1, graph2 = st.columns(2)

with graph1:

    if os.path.exists("accuracy.png"):

        st.subheader("Accuracy Graph")

        st.image(
            "accuracy.png",
            use_container_width=True
        )

with graph2:

    if os.path.exists("loss.png"):

        st.subheader("Loss Graph")

        st.image(
            "loss.png",
            use_container_width=True
        )


if os.path.exists("confusion_matrix.png"):

    st.subheader("Confusion Matrix")

    st.image(
        "confusion_matrix.png",
        use_container_width=True
    )


if os.path.exists("class_distribution.png"):

    st.subheader("Dataset Class Distribution")

    st.image(
        "class_distribution.png",
        use_container_width=True
    )


# ----------------------------------
# CLASS INFORMATION
# ----------------------------------

st.markdown("---")

st.header("Understanding the Seven Classes")

for category in class_names:

    description = class_descriptions.get(
        category,
        "Skin lesion category"
    )

    st.write(
        f"**{category.upper()}** — {description}"
    )


# ----------------------------------
# FOOTER
# ----------------------------------

st.markdown("---")

st.info(
    "Disclaimer: SkinSight AI is an academic research "
    "prototype. It must not be used for self-diagnosis, "
    "treatment decisions, or replacing professional "
    "medical advice."
)

st.caption(
    "SkinSight AI | Deep Learning Semester Project"
)
