import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# ----------------------------
# Load Model
# ----------------------------
model_path = "trained_model (1).sav"

if os.path.exists(model_path):
    model = pickle.load(open(model_path, "rb"))
else:
    st.error("Model file not found!")
    st.stop()

# ----------------------------
# App Title
# ----------------------------
st.title("Machine Learning Prediction App")
st.write("Enter the feature values below to get prediction.")

# ----------------------------
# Input Fields (Example: 4 features)
# Change number of features according to your model
# ----------------------------
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
feature3 = st.number_input("Feature 3")
feature4 = st.number_input("Feature 4")

# ----------------------------
# Prediction Button
# ----------------------------
if st.button("Predict"):
    
    # Convert inputs into numpy array
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Show result
    st.success(f"Prediction Result: {prediction[0]}")
