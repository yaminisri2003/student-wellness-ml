import pandas as pd

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def split_features_target(self):
        X = self.df.drop("Wellness_Level", axis=1)
        y = self.df["Wellness_Level"]
        return X, y

    def encode_features(self, X):
        X = pd.get_dummies(X, drop_first=True)
        print("Categorical features encoded")
        return X






