import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os

# ----------------------------
# Load Model
# ----------------------------
model_path = "model (2).pkl"   # Make sure this file is in same folder

if os.path.exists(model_path):
    with open(model_path, "rb") as file:
        model = pickle.load(file)
else:
    st.error("Model file not found. Please check the file name.")
    st.stop()

# ----------------------------
# App Title
# ----------------------------
st.title("Machine Learning Prediction App")
st.write("Enter input feature values below:")

# ----------------------------
# Get number of features
# ----------------------------
try:
    n_features = model.n_features_in_
except:
    st.error("Cannot detect number of features.")
    st.stop()

# ----------------------------
# Dynamic Input Fields
# ----------------------------
input_data = []

for i in range(n_features):
    value = st.number_input(f"Feature {i+1}", step=0.1)
    input_data.append(value)

# ----------------------------
# Prediction
# ----------------------------
if st.button("Predict"):
    
    # Convert to numpy array
    input_array = np.array([input_data])
    
    # Make prediction
    prediction = model.predict(input_array)
    
    st.success(f"Prediction Result: {prediction[0]}")
