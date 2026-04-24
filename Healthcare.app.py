import streamlit as st
import joblib
import numpy as np

# 1. Load the trained model
model = joblib.load('best_skin_disorder_model.pkl')

st.title("Skin Disorder Prediction App")
st.write("Enter the patient details below to predict the disorder type.")

# 2. Create input fields (Adjust these based on your actual features)
# Based on your notebook, ensure these match the features used in X_train
feature1 = st.number_input("Feature 1 (e.g., Age)", value=0.0)
feature2 = st.number_input("Feature 2", value=0.0)
feature3 = st.number_input("Feature 3", value=0.0)
feature4 = st.number_input("Feature 4", value=0.0)
# Add more st.number_input fields as needed to match your model's expected input

if st.button("Predict"):
    # 3. Prepare the data for prediction
    input_data = np.array([[feature1, feature2, feature3, feature4]])
    
    # 4. Make prediction
    prediction = model.predict(input_data)
    
    st.success(f"The predicted Skin Disorder Class is: {prediction[0]}")
