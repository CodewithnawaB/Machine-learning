import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# -----------------------------
# Load and preprocess data
# -----------------------------
@st.cache_data
def load_and_train_model():
    data = pd.read_csv(r'C:\mechine_learning\Sleep_health_and_lifestyle_dataset.csv')
    data[['Systolic BP', 'Diastolic BP']] = data['Blood Pressure'].str.split('/', expand=True).astype(int)
    data.drop('Blood Pressure', axis=1, inplace=True)
    
    X = data[['Age', 'Stress Level', 'Physical Activity Level']]
    y = data['Sleep Duration']
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.values.reshape(-1, 1))
    
    model = LinearRegression()
    model.fit(X_scaled, y_scaled)
    return model, scaler_X, scaler_y

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("😴 Sleep Duration Predictor")
st.markdown("Predict your average sleep duration (hours) based on age, stress level, and activity level.")

# Get model and scalers
model, scaler_X, scaler_y = load_and_train_model()

# Input fields
age = st.slider("Age (years)", 18, 80, 30)
stress = st.slider("Stress Level (1-10)", 1, 10, 5)
activity = st.slider("Physical Activity Level (minutes per day)", 0, 120, 30)

# Predict
if st.button("Predict Sleep Duration"):
    new_data = pd.DataFrame({'Age': [age], 'Stress Level': [stress], 'Physical Activity Level': [activity]})
    scaled = scaler_X.transform(new_data)
    pred_scaled = model.predict(scaled)
    pred_original = scaler_y.inverse_transform(pred_scaled)
    st.success(f"🕒 Estimated Sleep Duration: **{pred_original[0][0]:.2f} hours**")

st.caption("Built with Streamlit + Scikit-learn.")
