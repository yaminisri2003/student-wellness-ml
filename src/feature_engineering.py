import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

class FeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()

    def encode_target(self):
        self.df["Wellness_Level"] = self.label_encoder.fit_transform(
            self.df["Wellness_Level"]
        )
        print("Target encoded")
        return self.df

    def split_features_target(self):
        X = self.df.drop(columns=["Wellness_Level"])
        y = self.df["Wellness_Level"]
        return X, y

    def encode_categorical_features(self, X):
        X_encoded = pd.get_dummies(X, drop_first=True)
        print("Categorical features encoded")
        return X_encoded

    def scale_features(self, X):
        X_scaled = self.scaler.fit_transform(X)
        print("Features scaled")
        return X_scaled

    def prepare_data(self):
        self.encode_target()
        X, y = self.split_features_target()
        X = self.encode_categorical_features(X)
        X = self.scale_features(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print("Train-Test Split Done")
        print("Train shape:", X_train.shape)
        print("Test shape:", X_test.shape)

        return X_train, X_test, y_train, y_test
