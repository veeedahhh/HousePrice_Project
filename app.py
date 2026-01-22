import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model
try:
    model = joblib.load("model/house_price_model.pkl")  # adjust if needed
    st.write("✅ Model loaded successfully!")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# Streamlit UI
st.title("🏠 House Price Prediction System")
st.write("Predict house prices based on selected features.")

# User inputs
OverallQual = st.slider("Overall Quality (1-10)", 1, 10, 5)
GrLivArea = st.number_input("Above Ground Living Area (sq ft)", min_value=300, max_value=10000, value=1500)
TotalBsmtSF = st.number_input("Total Basement Area (sq ft)", min_value=0, max_value=5000, value=800)
GarageCars = st.slider("Number of Garage Cars", 0, 5, 1)
FullBath = st.slider("Number of Full Bathrooms", 0, 5, 1)
YearBuilt = st.number_input("Year Built", min_value=1800, max_value=2026, value=2000)

# Create input dataframe
input_data = pd.DataFrame({
    'OverallQual': [int(OverallQual)],
    'GrLivArea': [float(GrLivArea)],
    'TotalBsmtSF': [float(TotalBsmtSF)],
    'GarageCars': [int(GarageCars)],
    'FullBath': [int(FullBath)],
    'YearBuilt': [int(YearBuilt)]
})

# Ensure correct column order
input_data = input_data[['OverallQual', 'GrLivArea', 'TotalBsmtSF', 'GarageCars', 'FullBath', 'YearBuilt']]

# Convert types
input_data = input_data.astype(float)

# Show input data (for debugging)
st.write("Input Data:")
st.dataframe(input_data)

# Prediction
if st.button("Predict House Price"):
    try:
        prediction = model.predict(input_data)[0]
        st.success(f"🏡 Estimated House Price: ${prediction:,.2f}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
