from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier


class AdvancedModels:

    @staticmethod
    def get_models(preprocessor):

        models = {}

        # 1️⃣ Logistic Regression (Baseline)
        models["Logistic Regression"] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            ))
        ])

        # 2️⃣ Random Forest
        models["Random Forest"] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                class_weight="balanced"
            ))
        ])

        # 3️⃣ Gradient Boosting
        models["Gradient Boosting"] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", GradientBoostingClassifier(
                n_estimators=200,
                random_state=42
            ))
        ])

        # 4️⃣ XGBoost
        models["XGBoost"] = Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                use_label_encoder=False,
                eval_metric="logloss"
            ))
        ])

        return models

