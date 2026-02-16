import pandas as pd

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self) -> pd.DataFrame:
        try:
            df = pd.read_csv(self.file_path)
            print("Data loaded successfully")
            print("Shape:", df.shape)
            return df
        except Exception as e:
            print("Error loading data:", e)
            raise
