from src.data_loader import DataLoader
from src.data_cleaner import DataCleaner
from src.feature_engineering import FeatureEngineer

RAW_PATH = "data/raw/students_mental_health.csv"
PROCESSED_PATH = "data/processed/students_cleaned.csv"

def main():
    # Load Data
    loader = DataLoader(RAW_PATH)
    df = loader.load_data()

    # Clean Data
    cleaner = DataCleaner(df)
    cleaned_df = cleaner.clean_data()

    # Save cleaned data
    cleaned_df.to_csv(PROCESSED_PATH, index=False)
    print("Cleaned dataset saved successfully!")

    # Feature Engineering
    engineer = FeatureEngineer(cleaned_df)
    X_train, X_test, y_train, y_test = engineer.prepare_data()

if __name__ == "__main__":
    main()

