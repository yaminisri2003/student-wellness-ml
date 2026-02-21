# src/feature_importance.py

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, PartialDependenceDisplay


class FeatureImportance:

    def __init__(self, model, X_train, y_train):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train

    # -------------------------------------------------
    # ✅ Permutation Importance
    # -------------------------------------------------
    def compute_permutation_importance(self):
        print("\n🔍 Computing Permutation Feature Importance...")

        result = permutation_importance(
            self.model,
            self.X_train,
            self.y_train,
            n_repeats=10,
            random_state=42,
            scoring="f1"
        )

        importance_df = pd.DataFrame({
            "feature": self.X_train.columns,
            "importance": result.importances_mean
        }).sort_values(by="importance", ascending=False)

        os.makedirs("artifacts", exist_ok=True)

        # Save CSV
        importance_df.to_csv("artifacts/permutation_importance.csv", index=False)

        # Plot Top 15
        top_features = importance_df.head(10)

        plt.figure(figsize=(8, 6))
        plt.barh(
            top_features["feature"][::-1],
            top_features["importance"][::-1]
        )
        plt.title("Top 10 Permutation Feature Importance")
        plt.xlabel("Importance")
        plt.tight_layout()

        # ✅ Save image directly inside artifacts
        plt.savefig("artifacts/permutation_importance.png")
        plt.close()

        print("✅ Permutation importance saved -> artifacts/")
        return importance_df

    # -------------------------------------------------
    # ✅ Partial Dependence Plots (FIXED)
    # -------------------------------------------------
    def plot_pdp(self, features=None):
        print("\n📊 Plotting Partial Dependence Plots...")

        if features is None:
            features = self.X_train.columns[:5]

        os.makedirs("artifacts", exist_ok=True)

        for feature in features:
            try:
                disp = PartialDependenceDisplay.from_estimator(
                    self.model,
                    self.X_train,
                    features=[feature],
                    kind="average"
                )

                # ✅ Save PDP directly as image
                plt.gcf()
                plt.tight_layout()

                filename = f"artifacts/pdp_{feature}.png"
                plt.savefig(filename)
                plt.close()

                print(f"✅ Saved PDP for {feature}")

            except Exception as e:
                print(f"⚠ Error plotting PDP for {feature}: {e}")

        print("✅ All PDP plots saved inside artifacts/")