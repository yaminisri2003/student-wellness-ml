# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os

# -------------------------------
# Streamlit App Title
# -------------------------------
st.set_page_config(page_title="Student Depression ML Dashboard", layout="wide")
st.title("📊 Student Depression Project Dashboard")

# -------------------------------
# 1️⃣ Load Data
# -------------------------------
st.header("Dataset Overview")
try:
    df = pd.read_csv("data/raw/student_depression_dataset.csv")
    st.write(f"Dataset Shape: {df.shape}")
    st.dataframe(df.head(10))
except FileNotFoundError:
    st.error("Dataset not found. Make sure 'data/raw/student_depression_dataset.csv' exists.")

# -------------------------------
# 2️⃣ Load Test Metrics
# -------------------------------
st.header("Test Set Metrics")
try:
    metrics_df = pd.read_csv("artifacts/test_metrics.csv")
    st.write(metrics_df)
except FileNotFoundError:
    st.warning("Test metrics not found. Run main.py first.")

# -------------------------------
# 3️⃣ Confusion Matrix
# -------------------------------
st.header("Confusion Matrix")
try:
    pred_df = pd.read_csv("artifacts/test_predictions.csv")
    from sklearn.metrics import confusion_matrix
    import numpy as np

    cm = confusion_matrix(pred_df["y_true"], pred_df["y_pred"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)
except FileNotFoundError:
    st.warning("Test predictions not found. Run main.py first.")

# -------------------------------
# 4️⃣ Feature Importance
# -------------------------------
st.header("Feature Importance")

fi_folder = "artifacts/"

try:
    fi_files = [
        f for f in os.listdir(fi_folder)
        if ("permutation_importance" in f or "pdp_" in f)
        and f.endswith((".png"))
    ]

    if fi_files:
        st.write("Permutation Importance & PDP Plots:")

        for f in fi_files:
            file_path = os.path.join(fi_folder, f)

            # ✅ Extra safety — avoid folders
            if os.path.isfile(file_path):
                img = Image.open(file_path)
                st.image(img, caption=f, width=700)

    else:
        st.warning("Feature importance plots not found.")

except FileNotFoundError:
    st.warning("Artifacts folder missing. Run main.py first.")

# -------------------------------
# 5️⃣ SHAP Summary
# -------------------------------
st.header("SHAP Explainability")
shap_folder = "artifacts/shap"
try:
    shap_files = [f for f in os.listdir(shap_folder) if f.endswith(".png")]
    if shap_files:
        for f in shap_files:
            img = Image.open(os.path.join(shap_folder, f))
            st.image(img, caption=f, width=700)
    else:
        st.warning("SHAP plots not found.")
except FileNotFoundError:
    st.warning("SHAP folder missing. Run main.py first.")