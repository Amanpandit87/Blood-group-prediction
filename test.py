import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- Page Configuration ---
st.set_page_config(page_title="Fingerprint Blood Group Detector", layout="centered")

st.title("🩸 Fingerprint Blood Group Detector")
st.write("Bhai, apni fingerprint image upload karo aur dekho model kya kehta hai!")

# --- Step 1: Model Load Karein ---
model_path = 'blood_group_model.h5'

@st.cache_resource # Isse model baar-baar load nahi hoga, app fast chalegi
def load_my_model():
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
    return None

model = load_my_model()

if model is None:
    st.error(f"Bhai, '{model_path}' file nahi mili! Pehle model save kar lo.")
    st.stop()

# --- Step 2: Categories ---
class_names = ['A-', 'A+', 'AB-', 'AB+', 'B-', 'B+', 'O-', 'O+']

# --- Step 3: File Uploader ---
# .bmp files include kar li hain
uploaded_file = st.file_uploader("Fingerprint image select karein...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # Image load karein
    img = Image.open(uploaded_file)
    
    # --- FIX: Convert to RGB (4 channels ko 3 mein badalne ke liye) ---
    img = img.convert('RGB')
    
    st.image(img, caption='Uploaded Fingerprint', use_column_width=True)
    
    # --- Step 4: Preprocessing ---
    st.write("Analysis chal raha hai...")
    
    # Resize aur Array mein convert
    img_height, img_width = 180, 180
    img_resized = img.resize((img_width, img_height))
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = tf.expand_dims(img_array, 0) # Batch axis (1, 180, 180, 3)
    
    # Normalization (Wahi jo training mein kiya tha)
    img_array = img_array / 255.0

    # --- Step 5: Prediction ---
    predictions = model.predict(img_array)
    
    # Agar model output logits de raha hai toh softmax lagayein, 
    # varna seedha probability use karein.
    score = tf.nn.softmax(predictions[0]).numpy()
    
    predicted_index = np.argmax(score)
    predicted_class = class_names[predicted_index]
    confidence = 100 * score[predicted_index]

    # --- Step 6: Results Display ---
    st.divider()
    st.success(f"### RESULT: Ye fingerprint **{predicted_class}** Blood Group ka hai!")
    
    # Confidence bar
    st.write(f"Model Confidence: **{confidence:.2f}%**")
    st.progress(int(confidence))

    # All Probabilities expander
    with st.expander("Sari categories ka confidence score dekhein"):
        for i in range(len(class_names)):
            col1, col2 = st.columns([1, 4])
            col1.write(f"**{class_names[i]}**")
            col2.progress(int(score[i] * 100))
            st.write(f"Score: {100 * score[i]:.2f}%")