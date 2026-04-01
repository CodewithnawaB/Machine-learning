# ================================
# 🌍 Life Expectancy Prediction App
# ================================

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# -------------------------------
# App Header
# -------------------------------
st.set_page_config(page_title="Life Expectancy Prediction", layout="wide")
st.title("🌍 Life Expectancy Prediction using GDP and Year")
st.write("Predict schooling years based on GDP and Year using a Linear Regression model.")

# -------------------------------
# Upload or use default dataset
# -------------------------------
uploaded_file = st.file_uploader("C:\mechine_learning\LifeExpectancy.csv", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
else:
    st.info("Using sample dataset (LifeExpectancy.csv in project folder).")
    data = pd.read_csv("LifeExpectancy.csv")

# Clean data
data = data.dropna()
data.columns = data.columns.str.strip()
selected_data = data[['Year', 'GDP', 'Schooling']].dropna()

# Train model
X = selected_data[['Year', 'GDP']]
y = selected_data['Schooling']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
st.success(f"✅ Model trained successfully! R² Score: {r2:.3f}, MSE: {mse:.3f}")

# -------------------------------
# Prediction Section
# -------------------------------
st.header("🎯 Predict Schooling Years")

year = st.number_input("Enter Year:", min_value=1900, max_value=2100, value=2025)
gdp = st.number_input("Enter GDP per Capita:", min_value=0.0, value=1500.0)

if st.button("Predict"):
    new_data = pd.DataFrame({'Year': [year], 'GDP': [gdp]})
    predicted = model.predict(new_data)[0]
    st.subheader(f"📘 Predicted Schooling for {year}: **{predicted:.2f} years**")

    # Visualization
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(selected_data['GDP'], selected_data['Schooling'], color='blue', alpha=0.5, label='Actual Data')
    ax.scatter(gdp, predicted, color='red', s=200, marker='*', label='Predicted Point')
    ax.set_xlabel('GDP per Capita')
    ax.set_ylabel('Schooling (Years)')
    ax.set_title('GDP vs Schooling with Prediction')
    ax.legend()
    st.pyplot(fig)

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("📊 Built with Streamlit | Linear Regression Example")
