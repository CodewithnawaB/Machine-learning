# ======================================================
# Streamlit App — California Housing Price Prediction (3D)
# Manual Multiple Linear Regression (Normal Equation)
# ======================================================
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting

st.set_page_config(page_title="3D California Housing Predictor", page_icon="🏠", layout="centered")

st.title("🏠 California Housing Price Prediction — 3D Visualization")
st.write("""
This app uses **Multiple Linear Regression (Normal Equation)** on the 
California Housing dataset to predict **Median House Value**.
It also displays a **3D regression surface** between two main features.
""")

# -------- Step 1: Load Dataset Automatically --------
DATA_PATH = r"C:\mechine_learning\housing.csv"  # use raw string to avoid escape issues

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset file `housing.csv` not found in the directory!")
    st.stop()

data = pd.read_csv(DATA_PATH)
st.success("✅ Dataset Loaded Successfully!")

# -------- Step 2: Select Important Features --------
features = ["median_income", "longitude", "latitude", "housing_median_age", "population"]
target = "median_house_value"

X = data[features].values
y = data[target].values
m = len(y)

# -------- Step 3: Normalize Features --------
mu = X.mean(axis=0)
sigma = X.std(axis=0)
X_norm = (X - mu) / sigma
X_b = np.c_[np.ones((m, 1)), X_norm]

# -------- Step 4: Compute θ using Normal Equation --------
theta = np.linalg.pinv(X_b.T @ X_b) @ X_b.T @ y
st.subheader("📊 Model Parameters (θ Values)")
st.dataframe(pd.DataFrame(theta, index=["Bias"] + features, columns=["Theta Value"]))

# -------- Step 5: Input Fields for Prediction --------
st.subheader("🧮 Enter House Details for Prediction")

median_income = st.number_input("Median Income", value=3.5, step=0.1)
longitude = st.number_input("Longitude", value=-122.0, step=0.1)
latitude = st.number_input("Latitude", value=37.0, step=0.1)
housing_median_age = st.number_input("Housing Median Age (years)", value=20.0, step=1.0)
population = st.number_input("Population", value=1500.0, step=100.0)

# Normalize input
X_user = np.array([[median_income, longitude, latitude, housing_median_age, population]])
X_user_norm = (X_user - mu) / sigma
X_user_bias = np.c_[np.ones((1, 1)), X_user_norm]

# Predict
predicted_value = X_user_bias.dot(theta)[0]
st.success(f"🏡 **Predicted Median House Value:** ${predicted_value:,.2f}")

# -------- Step 6: 3D Visualization --------
st.subheader("🌐 3D Visualization: Regression Surface")

# Choose two features for 3D view
x1_name = "median_income"
x2_name = "housing_median_age"

x1 = data[x1_name]
x2 = data[x2_name]
y_true = y

# Build grid for surface
x1_range = np.linspace(x1.min(), x1.max(), 40)
x2_range = np.linspace(x2.min(), x2.max(), 40)
X1, X2 = np.meshgrid(x1_range, x2_range)

# Normalize based on training normalization
x1_mu, x1_sigma = mu[0], sigma[0]  # median_income
x2_mu, x2_sigma = mu[3], sigma[3]  # housing_median_age
X1_norm = (X1 - x1_mu) / x1_sigma
X2_norm = (X2 - x2_mu) / x2_sigma

# Keep other features fixed at mean (longitude, latitude, population)
Z = (theta[0]
     + theta[1]*X1_norm                 # median_income
     + theta[4]*X2_norm                 # housing_median_age
     + theta[2]*0                       # longitude mean effect
     + theta[3]*0                       # latitude mean effect
     + theta[5]*0)                      # population mean effect

# Create 3D Figure
fig = plt.figure(figsize=(10,7))
ax = fig.add_subplot(111, projection="3d")

# Plot actual data
ax.scatter(x1, x2, y_true, color='blue', alpha=0.4, label='Actual Data')

# Plot regression surface
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.6, edgecolor='none')

ax.set_xlabel("Median Income")
ax.set_ylabel("Housing Median Age")
ax.set_zlabel("Median House Value ($)")
ax.set_title("3D Regression Surface (Normal Equation)")
plt.legend()
st.pyplot(fig)
