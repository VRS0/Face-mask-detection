import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Load model
model = load_model("Face_mask_classifier_model.h5")

# Title and Subtitle (Centered with style)
st.markdown("""
    <div style='text-align: center;'>
        <h1 style='color:purple;'>~ Face Mask Detection 😷 ~</h1>
        <p style='font-size: 14px; color: gray;'>Made by Abdullah Haitham</p>
    </div>
""", unsafe_allow_html=True)

# Image uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)

    # Predict
    prediction = model.predict(image)[0][0]

    # Confidence and label
    if prediction >= 0.5:
        label = "With Mask ✅"
        confidence = prediction
        st.success(f"**{label}**")
    else:
        label = "Without Mask ❌"
        confidence = 1 - prediction
        st.error(f"**{label}**")

    st.info(f"**Confidence:** {confidence:.2%}")
