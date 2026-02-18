import joblib
import os
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import seaborn as sns
import matplotlib.pyplot as plt


class ModelTrainer:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        # Ensure models directory exists
        os.makedirs("models", exist_ok=True)

    # ----------------------------
    # 1. Train-Test Split
    # ----------------------------
    def split_data(self):
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            test_size=0.2,
            random_state=42,
            stratify=self.y  # important for imbalance
        )

        print("Train-Test Split Done")
        print("Train shape:", X_train.shape)
        print("Test shape:", X_test.shape)

        return X_train, X_test, y_train, y_test

    # ----------------------------
    # 2. Feature Scaling
    # ----------------------------
    def scale_features(self, X_train, X_test):
        scaler = StandardScaler()

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        print("Features scaled")

        # Save scaler for future inference
        joblib.dump(scaler, "models/scaler.pkl")

        return X_train, X_test

    # ----------------------------
    # 3. Logistic Regression
    # ----------------------------
    def train_logistic_regression(self, X_train, X_test, y_train, y_test):

        print("\nTraining Logistic Regression (Balanced)...")

        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        print("\nLogistic Regression Results")
        print("Accuracy:", round(accuracy, 4))
        print(report)

        # Extract F1 for Moderate Wellness
        if "Moderate Wellness" in report_dict:
            moderate_f1 = report_dict["Moderate Wellness"]["f1-score"]
            moderate_recall = report_dict["Moderate Wellness"]["recall"]

            print("F1 Score (Moderate Wellness):", round(moderate_f1, 4))
            print("Recall (Moderate Wellness):", round(moderate_recall, 4))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        print("\nConfusion Matrix:")
        print(cm)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix - Logistic Regression")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.show()

        joblib.dump(model, "models/logistic_regression.pkl")
        print("Logistic Regression model saved")

    # ----------------------------
    # 4. Random Forest
    # ----------------------------
    def train_random_forest(self, X_train, X_test, y_train, y_test):

        print("\nTraining Random Forest Classifier (Balanced)...")

        model = RandomForestClassifier(
            random_state=42,
            class_weight="balanced"
        )

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        report_dict = classification_report(y_test, y_pred, output_dict=True)

        print("\nRandom Forest Results")
        print("Accuracy:", round(accuracy, 4))
        print(report)

        # Extract F1 for Moderate Wellness
        if "Moderate Wellness" in report_dict:
            moderate_f1 = report_dict["Moderate Wellness"]["f1-score"]
            moderate_recall = report_dict["Moderate Wellness"]["recall"]

            print("F1 Score (Moderate Wellness):", round(moderate_f1, 4))
            print("Recall (Moderate Wellness):", round(moderate_recall, 4))

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)

        print("\nConfusion Matrix:")
        print(cm)

        plt.figure()
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix - Random Forest")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.show()

        joblib.dump(model, "models/random_forest_classifier.pkl")
        print("Random Forest model saved")



