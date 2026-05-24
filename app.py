"""
AI Image Detector - Streamlit Web App
Upload an image and get instant predictions!
"""

import streamlit as st
import numpy as np
import json
import os
from PIL import Image
import tensorflow as tf
from tensorflow import keras

st.set_page_config(
    page_title="AI Image Detector",
    page_icon="🤖",
    layout="centered"
)

MODEL_PATH = 'outputs/best_model.keras'
CLASS_INDICES_PATH = 'outputs/class_indices.json'
IMG_SIZE = (224, 224)

@st.cache_resource
def load_model():
    try:
        model = keras.models.load_model(MODEL_PATH)
        
        with open(CLASS_INDICES_PATH, 'r') as f:
            class_indices = json.load(f)
        
        class_names = {v: k for k, v in class_indices.items()}
        
        return model, class_names
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("Please make sure you've trained the model first using 'python train.py'")
        return None, None

def preprocess_image(image):
    image = image.resize(IMG_SIZE)
    image_array = np.array(image)
    
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=-1)
    elif image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]
    
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_image(model, image, class_names):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image, verbose=0)
    
    predicted_index = np.argmax(predictions[0])
    predicted_class = class_names[predicted_index]
    confidence = predictions[0][predicted_index] * 100
    
    all_probabilities = {
        class_names[i]: float(predictions[0][i] * 100)
        for i in range(len(class_names))
    }
    
    return predicted_class, confidence, all_probabilities

def main():
    st.title("🤖 AI Image Detector")
    st.markdown("### Upload an image to detect if it's AI-generated or a real photograph")
    st.markdown("---")
    
    with st.spinner("Loading model..."):
        model, class_names = load_model()
    
    if model is None:
        st.stop()
    
    st.success("✅ Model loaded successfully!")
    
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a clear image for best results"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Original Image")
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with st.spinner("🔍 Analyzing image..."):
            predicted_class, confidence, all_probabilities = predict_image(
                model, image, class_names
            )
        
        with col2:
            st.markdown("#### Prediction Results")
            
            if predicted_class.lower() == 'ai':
                st.error(f"🤖 **AI-Generated**")
            else:
                st.success(f"📷 **Real Photograph**")
            
            st.metric(
                label="Confidence Score",
                value=f"{confidence:.2f}%"
            )
            
            st.progress(confidence / 100)
            
            st.markdown("---")
            st.markdown("#### Interpretation")
            if confidence >= 90:
                st.info("🎯 **Very High Confidence** - The model is very certain about this prediction.")
            elif confidence >= 70:
                st.info("✅ **High Confidence** - The model is fairly certain about this prediction.")
            elif confidence >= 50:
                st.warning("⚠️ **Moderate Confidence** - The model has some uncertainty.")
            else:
                st.warning("⚠️ **Low Confidence** - The model is uncertain. Results may be unreliable.")
        
        st.markdown("---")
        st.markdown("### 📊 Detailed Probability Breakdown")
        
        for class_name, probability in sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True):
            st.write(f"**{class_name.upper()}:** {probability:.2f}%")
            st.progress(probability / 100)
    
    else:
        st.info("👆 Upload an image to get started!")
        
        st.markdown("---")
        st.markdown("### 📝 How to Use")
        st.markdown("""
        1. **Upload** an image using the file uploader above
        2. **Wait** for the model to analyze the image
        3. **View** the prediction and confidence score
        4. **Interpret** the results based on the confidence level
        """)
        
        st.markdown("### ℹ️ About This Project")
        st.markdown("""
        This AI Image Detector uses a deep learning model trained with transfer learning 
        (MobileNetV2) to classify images as either AI-generated or real photographs.
        
        **Features:**
        - ✅ Fast inference (< 1 second)
        - ✅ High accuracy with confidence scores
        - ✅ Easy-to-use web interface
        - ✅ Detailed probability breakdown
        """)
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center'>"
        "<p>Built with ❤️ using TensorFlow & Streamlit</p>"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()