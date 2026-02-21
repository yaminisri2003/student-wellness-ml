import os
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from sklearn.model_selection import cross_val_score, StratifiedKFold


class ModelEvaluator:

    @staticmethod
    def cross_validate_model(model, X, y, scoring="f1"):
        """
        Perform 5-fold Stratified Cross Validation.
        Returns mean CV score.
        """

        print("\n🔎 Running 5-Fold Cross-Validation...")

        cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )

        scores = cross_val_score(
            model,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=-1
        )

        print(f"Scores per fold : {scores}")
        print(f"Mean CV Score   : {scores.mean():.4f}")
        print(f"Std Deviation   : {scores.std():.4f}")

        return scores.mean()


    @staticmethod
    def evaluate_models(models, X_train, X_test, y_train, y_test):
        """
        Train and evaluate multiple models on test data.
        Returns comparison dataframe.
        """

        results = []

        print("\n==============================")
        print(" MODEL EVALUATION STARTED")
        print("==============================")

        for name, model in models.items():

            print(f"\n🚀 Training {name}...")

            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            roc_auc = roc_auc_score(y_test, y_proba)

            results.append({
                "Model": name,
                "Accuracy": round(accuracy, 4),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1 Score": round(f1, 4),
                "ROC-AUC": round(roc_auc, 4)
            })

            print(f"{name} Completed ✔")

        results_df = pd.DataFrame(results).sort_values(
            by="F1 Score",
            ascending=False
        )

        # Save results
        os.makedirs("artifacts", exist_ok=True)
        results_df.to_csv("artifacts/model_comparison.csv", index=False)

        print("\n📊 Model Comparison (Sorted by F1):")
        print(results_df)

        print("\n✅ Results saved to artifacts/model_comparison.csv")

        return results_df




