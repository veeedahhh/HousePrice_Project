import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load("house_price_model.pkl")

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
    'OverallQual': [OverallQual],
    'GrLivArea': [GrLivArea],
    'TotalBsmtSF': [TotalBsmtSF],
    'GarageCars': [GarageCars],
    'FullBath': [FullBath],
    'YearBuilt': [YearBuilt]
})

# Prediction button
if st.button("Predict House Price"):
    prediction = model.predict(input_data)[0]
    st.success(f"Estimated House Price: ${prediction:,.2f}")
