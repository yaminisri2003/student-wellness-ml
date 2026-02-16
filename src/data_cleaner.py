import pandas as pd

class DataCleaner:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def handle_missing_values(self):
        # Fill CGPA with median
        self.df["CGPA"] = self.df["CGPA"].fillna(self.df["CGPA"].median())

        # Fill Substance_Use with mode
        self.df["Substance_Use"] = self.df["Substance_Use"].fillna(
            self.df["Substance_Use"].mode()[0]
        )

        return self.df

    def create_target_variable(self):
        # Create Mental Score
        self.df["Mental_Score"] = (
            self.df["Stress_Level"]
            + self.df["Depression_Score"]
            + self.df["Anxiety_Score"]
            + self.df["Financial_Stress"]
        )

        # Classification rule
        def classify_wellness(score):
            if score <= 5:
                return "High Wellness"
            elif score <= 12:
                return "Moderate Wellness"
            else:
                return "Low Wellness"

        self.df["Wellness_Level"] = self.df["Mental_Score"].apply(classify_wellness)

        return self.df

    def clean_data(self):
        self.handle_missing_values()
        self.create_target_variable()
        print(" Data cleaned successfully")
        return self.df
