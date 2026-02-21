# src/hyperparameter_tuning.py
import optuna
import os
import joblib
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier

class HyperparameterTuner:
    def __init__(self, X, y, n_splits=5, random_state=42):
        self.X = X
        self.y = y
        self.n_splits = n_splits
        self.random_state = random_state

    def objective(self, trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "random_state": self.random_state
        }

        model = GradientBoostingClassifier(**params)
        cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(model, self.X, self.y, cv=cv, scoring="f1")
        return scores.mean()

    def tune(self, n_trials=50):
        print("\n==============================")
        print(" HYPERPARAMETER TUNING (Gradient Boosting) ")
        print("==============================")

        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials, show_progress_bar=True)

        print("\n✅ Best Parameters Found:")
        print(study.best_params)
        print(f"Best CV F1 Score: {study.best_value:.4f}")

        # Save the study
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(study, "artifacts/gb_optuna_study.pkl")

        return study.best_params