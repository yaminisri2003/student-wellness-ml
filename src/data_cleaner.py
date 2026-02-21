from pathlib import Path
import pandas as pd

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    if 'id' in df.columns:
        df.drop(columns=['id'], inplace=True)

    df.columns = df.columns.str.strip()

    return df


def save_processed_data(df: pd.DataFrame, filename: str):
    """
    Save cleaned dataset to data/processed folder.
    """
    base_path = Path(__file__).resolve().parent.parent
    save_path = base_path / "data" / "processed" / filename

    df.to_csv(save_path, index=False)

    print(f"Processed dataset saved to: {save_path}")


