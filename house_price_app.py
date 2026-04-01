# app.py
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# --- Page Setup ---
st.set_page_config(page_title="Car Price Predictor", layout="centered")

st.title("🚗 Automobile Price Predictor (Polynomial Regression)")
st.markdown("### Predict car prices based on engine size and fuel type")

# --- Load and prepare data ---
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/autos/imports-85.data"
cols = [
    "symboling", "normalized-losses", "make", "fuel-type", "aspiration", "num-of-doors",
    "body-style", "drive-wheels", "engine-location", "wheel-base", "length", "width",
    "height", "curb-weight", "engine-type", "num-of-cylinders", "engine-size",
    "fuel-system", "bore", "stroke", "compression-ratio", "horsepower",
    "peak-rpm", "city-mpg", "highway-mpg", "price"
]

df = pd.read_csv(url, names=cols, na_values='?')
df = df.dropna(subset=["engine-size", "price", "fuel-type"])
df["engine-size"] = df["engine-size"].astype(float)
df["price"] = df["price"].astype(float)
df = pd.get_dummies(df, columns=["fuel-type"], drop_first=True)

X = df[["engine-size", "fuel-type_diesel"]].values
y = df["price"].values

poly = PolynomialFeatures(degree=3, include_bias=False)
X_num = poly.fit_transform(X[:, [0]])
X_full = np.hstack([X_num, X[:, [1]]])

model = LinearRegression()
model.fit(X_full, y)

# --- Sidebar inputs ---
st.sidebar.header("🔧 Input Controls")
engine_size = st.sidebar.slider("Select Engine Size (cc)", 50, 350, 200, step=10)
fuel_type = st.sidebar.selectbox("Select Fuel Type", ["gas", "diesel"])

fuel_flag = 1 if fuel_type == "diesel" else 0
X_pred = np.hstack([poly.transform([[engine_size]]), [[fuel_flag]]])
predicted_price = model.predict(X_pred)[0]

# --- Display prediction ---
st.subheader("💰 Predicted Price")
st.metric(label="Estimated Car Price ($)", value=f"{predicted_price:,.2f}")

# --- Visualization ---
x_seq = np.linspace(df["engine-size"].min(), df["engine-size"].max(), 200).reshape(-1, 1)
x_poly = poly.transform(x_seq)
y_gas = model.predict(np.hstack([x_poly, np.zeros((200, 1))]))
y_diesel = model.predict(np.hstack([x_poly, np.ones((200, 1))]))

fig, ax = plt.subplots(figsize=(7,4))
ax.plot(x_seq, y_gas, color='red', label="Gas")
ax.plot(x_seq, y_diesel, color='blue', label="Diesel")
ax.scatter(df["engine-size"], df["price"], color='gray', alpha=0.5, s=10)
ax.axvline(engine_size, color='green', linestyle='--', label="Selected Engine Size")
ax.set_xlabel("Engine Size (cc)")
ax.set_ylabel("Price ($)")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.caption("Developed with ❤️ using Streamlit and Scikit-Learn")
