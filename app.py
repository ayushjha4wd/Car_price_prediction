import streamlit as st
import numpy as np
import joblib
import os

# ----------------------------
# Load Model
# ----------------------------
model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

if os.path.exists(model_path):
    model = joblib.load(model_path)
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
    input_array = np.array([input_data])
    prediction = model.predict(input_array)
    st.success(f"Prediction Result: {prediction[0]}")
