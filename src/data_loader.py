from pathlib import Path
import pandas as pd

def load_data(filename: str) -> pd.DataFrame:
    """
    Load dataset from data/raw folder.
    """
    base_path = Path(__file__).resolve().parent.parent
    file_path = base_path / "data" / "raw" / "student_depression_dataset.csv"

    df = pd.read_csv(file_path)

    print(f"Dataset loaded successfully from: {file_path}")
    print(f"Shape: {df.shape}")

    return df
