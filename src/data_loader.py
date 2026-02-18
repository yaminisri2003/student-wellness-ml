import pandas as pd

class DataLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_data(self):
        df = pd.read_csv(self.file_path)
        print("Raw data loaded:", df.shape)
        return df

