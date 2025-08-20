import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load model and dataset
model = joblib.load("best_model.pkl")
df = pd.read_csv("processed.cleveland.data.txt")

# Clean dataset
df.replace('?', np.nan, inplace=True)
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

top_features = ['cp', 'thalach', 'oldpeak', 'ca', 'thal']

st.title("❤️ Heart Disease Prediction App")
st.write("An interactive dashboard for prediction and data insights")

# Sidebar inputs
st.sidebar.header("Patient Input Features")

def user_input():
    cp = st.sidebar.selectbox("Chest Pain Type (cp)", sorted(df['cp'].unique()))
    thalach = st.sidebar.slider("Max Heart Rate Achieved (thalach)", 
                                int(df['thalach'].min()), int(df['thalach'].max()), int(df['thalach'].mean()))
    oldpeak = st.sidebar.slider("ST Depression (oldpeak)", 
                                float(df['oldpeak'].min()), float(df['oldpeak'].max()), float(df['oldpeak'].mean()))
    ca = st.sidebar.selectbox("Number of Major Vessels (ca)", sorted(df['ca'].unique()))
    thal = st.sidebar.selectbox("Thalassemia (thal)", sorted(df['thal'].unique()))

    return pd.DataFrame([[cp, thalach, oldpeak, ca, thal]], columns=top_features)

input_df = user_input()

# Prediction
selected_input = input_df[top_features]
prediction = model.predict(selected_input)[0]
prediction_proba = model.predict_proba(selected_input)[0][1]

st.subheader("🔎 Prediction Result")
st.write("**Prediction:**", "💚 No Disease" if prediction == 0 else "❤️ Disease Detected")
st.progress(int(prediction_proba * 100))
st.write(f"**Probability of Disease:** {prediction_proba:.2f}")

# ---- Visualization Section ----

# 1. Feature Distributions
st.subheader("📊 Feature Distributions with Patient Value")
for feature in top_features:
    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, bins=20, ax=ax)
    ax.axvline(float(input_df[feature]), color='red', linestyle='--', label="Patient Value")
    ax.set_title(f"Distribution of {feature}")
    ax.legend()
    st.pyplot(fig)

# 2. Correlation Heatmap
st.subheader("🔗 Correlation Heatmap")
fig, ax = plt.subplots(figsize=(6, 4))
corr = df[top_features + ['target']].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# 3. ROC Curve (load from training step)
st.subheader("📈 Model ROC Curves")

roc_files = [
    ("PCA Dataset", "roc_curve_PCA_Dataset.png"),
    ("Feature-Selected Dataset", "roc_curve_Feature-Selected_Dataset.png")
]

for label, file in roc_files:
    try:
        st.image(file, caption=f"ROC Curve - {label}")
    except:
        st.warning(f"ROC curve for {label} not available. Please generate during training.")
