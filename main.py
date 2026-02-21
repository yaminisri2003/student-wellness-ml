# main.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap

from src.data_loader import load_data
from src.data_cleaner import clean_data
from src.hyperparameter_tuning import HyperparameterTuner

# -------------------------------
# Utility Functions
# -------------------------------
def save_test_metrics(y_true, y_pred, filepath="artifacts/test_metrics.csv"):
    metrics = {
        "Accuracy": [accuracy_score(y_true, y_pred)],
        "Precision": [precision_score(y_true, y_pred, zero_division=0)],
        "Recall": [recall_score(y_true, y_pred, zero_division=0)],
        "F1 Score": [f1_score(y_true, y_pred, zero_division=0)],
        "ROC-AUC": [roc_auc_score(y_true, y_pred)]
    }
    df = pd.DataFrame(metrics)
    df.to_csv(filepath, index=False)
    print(f"\n✅ Test metrics saved to {filepath}")


def save_test_predictions(y_true, y_pred, filepath="artifacts/test_predictions.csv"):
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df.to_csv(filepath, index=False)
    print(f"✅ Test predictions saved to {filepath}")


# -------------------------------
# Main
# -------------------------------
def main():
    print("\n==============================")
    print(" STUDENT DEPRESSION PROJECT ")
    print("==============================")

    # 1️⃣ Load & Clean Data
    df = load_data("data/raw/student_depression_dataset.csv")
    df = clean_data(df)
    df = df.drop("Have you ever had suicidal thoughts ?", axis=1)  # avoid leakage
    print(f"\n✅ Data Loaded & Cleaned | Shape: {df.shape}")

    # 2️⃣ Split Features & Target
    X = df.drop("Depression", axis=1)
    y = df["Depression"]

    categorical_cols = X.select_dtypes(include="object").columns.tolist()
    numerical_cols = X.select_dtypes(exclude="object").columns.tolist()

    # 3️⃣ Preprocessing Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )
    X_processed_array = preprocessor.fit_transform(X)
    ohe_cols = preprocessor.transformers_[1][1].get_feature_names_out(categorical_cols)
    all_cols = numerical_cols + ohe_cols.tolist()
    X_processed = pd.DataFrame(X_processed_array, columns=all_cols)
    print(f"\n✅ Preprocessing Done | Total Features: {X_processed.shape[1]}")

    # 4️⃣ Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, stratify=y, random_state=42
    )
    print("\n✅ Train/Test Split Completed")

    # 5️⃣ Hyperparameter Tuning (Gradient Boosting)
    print("\n🚀 Starting Hyperparameter Tuning (Gradient Boosting)")
    tuner = HyperparameterTuner(X_train, y_train)
    best_params = tuner.tune(n_trials=5)  # adjust n_trials as needed
    print(f"\n✅ Best Hyperparameters: {best_params}")

    # 6️⃣ Train Final Model
    final_model = GradientBoostingClassifier(**best_params, random_state=42)
    final_model.fit(X_train, y_train)
    print("\n✅ Final Gradient Boosting Model Trained")

    # 7️⃣ Evaluate on Test Set
    y_pred = final_model.predict(X_test)
    save_test_metrics(y_test, y_pred)
    save_test_predictions(y_test, y_pred)

    # -------------------------------
    # 8️⃣ Permutation Feature Importance
    # -------------------------------
    print("\n🔍 Computing Permutation Feature Importance...")
    os.makedirs("artifacts", exist_ok=True)
    perm_importance = permutation_importance(final_model, X_test, y_test, n_repeats=10, random_state=42)
    sorted_idx = perm_importance.importances_mean.argsort()[::-1]

    # Save bar plot
    plt.figure(figsize=(12, 8))
    sns.barplot(x=perm_importance.importances_mean[sorted_idx], y=X_test.columns[sorted_idx])
    plt.title("Permutation Feature Importance")
    plt.tight_layout()
    plt.savefig("artifacts/permutation_importance.png")
    plt.close()
    print("✅ Permutation Importance saved to artifacts/")

    # -------------------------------
    # 9️⃣ Partial Dependence Plots (top 5 features)
    # -------------------------------
    top_features = X_test.columns[sorted_idx[:5]]
    print(f"\n🔍 Generating PDP for top 5 features: {list(top_features)}")
    for feat in top_features:
        fig, ax = plt.subplots(figsize=(6, 4))
        PartialDependenceDisplay.from_estimator(final_model, X_test, [feat], ax=ax)
        plt.tight_layout()
        os.makedirs("artifacts/pdp", exist_ok=True)
        plt.savefig(f"artifacts/pdp/pdp_{feat}.png")
        plt.close()
    print("✅ PDP plots saved to artifacts/pdp/")

    # -------------------------------
    # 10️⃣ SHAP Explainability
    # -------------------------------
    print("\n🔍 Running SHAP Explainability...")
    os.makedirs("artifacts/shap", exist_ok=True)
    explainer = shap.Explainer(final_model, X_test)
    shap_values = explainer(X_test)

    # SHAP summary plot
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("artifacts/shap/shap_summary.png", bbox_inches='tight')
    plt.close()
    print("✅ SHAP summary plot saved to artifacts/shap/")

    print("\n🎯 All steps completed! Artifacts saved to 'artifacts/' for Streamlit dashboard.")


if __name__ == "__main__":
    main()