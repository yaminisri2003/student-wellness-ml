import pandas as pd

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def handle_missing_values(self):
        self.df.fillna(self.df.median(numeric_only=True), inplace=True)
        self.df.fillna(self.df.mode().iloc[0], inplace=True)
        return self.df

    def create_target_variable(self):
        def classify_wellness(score):
            if score <= 3:
                return "High Wellness"
            elif score <= 7:
                return "Moderate Wellness"
            else:
                return "Low Wellness"

        self.df["Wellness_Level"] = self.df["Depression_Score"].apply(classify_wellness)

        # Drop target source to avoid leakage
        self.df.drop("Depression_Score", axis=1, inplace=True)

        return self.df

    def clean_data(self):
        self.handle_missing_values()
        self.create_target_variable()
        print("Data cleaned successfully")
        return self.df

