import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report


class ModelTrainer:

    @staticmethod
    def train_logistic_regression(preprocessor, X_train, X_test, y_train, y_test):

        print("\n🚀 Training Logistic Regression Model...\n")

        # Create pipeline
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"   # handles imbalance
            ))
        ])

        # Train model
        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)

        # ---------------------------
        # Evaluation Metrics
        # ---------------------------
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        print("📊 Model Performance:")
        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")

        print("\n📄 Classification Report:\n")
        print(classification_report(y_test, y_pred, zero_division=0))

        # ---------------------------
        # Feature Importance
        # ---------------------------
        print("\n🔍 Extracting Feature Importance...\n")

        # Get feature names after preprocessing
        feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()

        # Get logistic regression coefficients
        coefficients = pipeline.named_steps["classifier"].coef_[0]

        # Create DataFrame
        feature_importance_df = pd.DataFrame({
            "feature": feature_names,
            "coefficient": coefficients,
            "importance": np.abs(coefficients)
        })

        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values(
            by="importance",
            ascending=False
        )

        # Save CSV
        os.makedirs("artifacts", exist_ok=True)
        feature_importance_df.to_csv(
            "artifacts/logistic_feature_importance.csv",
            index=False
        )

        print("✅ Feature importance saved to artifacts/logistic_feature_importance.csv")

        # ---------------------------
        # Plot Top 15 Features
        # ---------------------------
        top_features = feature_importance_df.head(15)

        plt.figure(figsize=(10, 6))
        plt.barh(top_features["feature"], top_features["importance"])
        plt.xlabel("Absolute Coefficient Value")
        plt.ylabel("Feature")
        plt.title("Top 15 Feature Importance - Logistic Regression")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("artifacts/top_15_feature_importance.png")
        plt.close()

        print("📊 Top 15 feature importance plot saved to artifacts/top_15_feature_importance.png")

        print("\n✅ Logistic Regression Training Complete!\n")

        return pipeline







