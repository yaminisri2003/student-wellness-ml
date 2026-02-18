import pandas as pd
from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.model_trainer import ModelTrainer
from src.eda import EDA


def main():

    # ----------------------------------
    # 1. Load Data
    # ----------------------------------
    loader = DataLoader("data/raw/students_mental_health.csv")
    df = loader.load_data()

    # ----------------------------------
    # 2. Clean Data
    # ----------------------------------
    cleaner = DataCleaner(df)
    df_cleaned = cleaner.clean_data()

    df_cleaned.to_csv("data/processed/cleaned_data.csv", index=False)
    print("Cleaned dataset saved successfully!")

    # ----------------------------------
    # 3. Run EDA (BEFORE encoding)
    # ----------------------------------
    eda = EDA(df_cleaned)
    eda.run_eda()

    # ----------------------------------
    # 4. Feature Engineering
    # ----------------------------------
    engineer = FeatureEngineer(df_cleaned)

    X, y = engineer.split_features_target()
    X = engineer.encode_features(X)

    # ----------------------------------
    # 5. Model Training
    # ----------------------------------
    trainer = ModelTrainer(X, y)

    X_train, X_test, y_train, y_test = trainer.split_data()
    X_train, X_test = trainer.scale_features(X_train, X_test)

    trainer.train_logistic_regression(X_train, X_test, y_train, y_test)
    trainer.train_random_forest(X_train, X_test, y_train, y_test)


if __name__ == "__main__":
    main()


