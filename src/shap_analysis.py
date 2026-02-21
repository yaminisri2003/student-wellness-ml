# -----------------------------
# shap_analysis.py
# -----------------------------
import os
import pandas as pd
import shap

class SHAPAnalysis:
    def __init__(self, model, X_train, preprocessor, numerical_cols, categorical_cols, sample_size=2000):
        """
        model: trained model (e.g., GradientBoostingClassifier)
        X_train: original training features (before preprocessing)
        preprocessor: fitted ColumnTransformer
        numerical_cols: list of numerical columns
        categorical_cols: list of categorical columns
        sample_size: number of rows to sample for SHAP (avoid huge computation)
        """
        self.model = model
        self.X_train = X_train
        self.preprocessor = preprocessor
        self.numerical_cols = numerical_cols
        self.categorical_cols = categorical_cols
        self.sample_size = sample_size

        # Prepare preprocessed features
        self.X_processed = preprocessor.transform(X_train)

        # Feature names after preprocessing
        cat_features = preprocessor.transformers_[1][1].get_feature_names_out(categorical_cols).tolist()
        self.feature_names = numerical_cols + cat_features

        # Convert to DataFrame
        self.X_processed_df = pd.DataFrame(self.X_processed, columns=self.feature_names)

        # Sample for SHAP
        if sample_size < self.X_processed_df.shape[0]:
            self.X_sample = self.X_processed_df.sample(sample_size, random_state=42)
        else:
            self.X_sample = self.X_processed_df

    def run_shap(self):
        # Create Tree Explainer
        explainer = shap.TreeExplainer(self.model)

        print("\n🔍 Running SHAP Explainability... (sampling {} rows)".format(self.X_sample.shape[0]))
        shap_values = explainer(self.X_sample, check_additivity=False)

        # Make output directory
        os.makedirs("artifacts/shap", exist_ok=True)

        # Summary plot
        print("📊 Creating summary plot...")
        shap.summary_plot(shap_values, self.X_sample, feature_names=self.feature_names, show=False)
        shap.summary_plot(shap_values, self.X_sample, feature_names=self.feature_names, plot_type="bar", show=False)
        print("✅ Summary plots saved in artifacts/shap/")

        # Save shap values for later use
        shap_values_array = shap_values.values if hasattr(shap_values, "values") else shap_values
        pd.DataFrame(shap_values_array, columns=self.feature_names).to_csv("artifacts/shap/shap_values.csv", index=False)
        print("✅ SHAP values saved to artifacts/shap/shap_values.csv")

        return shap_values
        