import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor


# -------------------------------------------------------
# MASTER FUNCTION: RUN ALL OUTLIER METHODS + COMPARISON
# -------------------------------------------------------
def run_outlier_analysis(df: pd.DataFrame, contamination=0.05):

    df = df.copy()
    numeric_cols = df.select_dtypes(include=np.number).columns
    total_rows = len(df)

    print("\n==============================")
    print(" OUTLIER ANALYSIS STARTED ")
    print("==============================\n")

    # -------------------------------------------------------
    # 1️⃣ IQR METHOD
    # -------------------------------------------------------
    iqr_flags = pd.Series(False, index=df.index)

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        col_outliers = (df[col] < lower) | (df[col] > upper)
        iqr_flags = iqr_flags | col_outliers

    iqr_count = iqr_flags.sum()
    iqr_percent = (iqr_count / total_rows) * 100

    # -------------------------------------------------------
    # 2️⃣ Z-SCORE METHOD
    # -------------------------------------------------------
    z_scores = np.abs(stats.zscore(df[numeric_cols]))
    z_flags = (z_scores > 3).any(axis=1)

    z_count = z_flags.sum()
    z_percent = (z_count / total_rows) * 100

    # -------------------------------------------------------
    # 3️⃣ Isolation Forest
    # -------------------------------------------------------
    iforest = IsolationForest(
        contamination=contamination,
        random_state=42
    )
    if_preds = iforest.fit_predict(df[numeric_cols])
    if_flags = if_preds == -1

    if_count = if_flags.sum()
    if_percent = (if_count / total_rows) * 100

    # -------------------------------------------------------
    # 4️⃣ Local Outlier Factor
    # -------------------------------------------------------
    lof = LocalOutlierFactor(n_neighbors=20)
    lof_preds = lof.fit_predict(df[numeric_cols])
    lof_flags = lof_preds == -1

    lof_count = lof_flags.sum()
    lof_percent = (lof_count / total_rows) * 100

    # -------------------------------------------------------
    # OVERLAP ANALYSIS (ML vs ML)
    # -------------------------------------------------------
    overlap_ml = (if_flags & lof_flags).sum()
    overlap_percent = (overlap_ml / total_rows) * 100

    # -------------------------------------------------------
    # SUMMARY TABLE
    # -------------------------------------------------------
    summary = pd.DataFrame({
        "Method": ["IQR", "Z-Score", "Isolation Forest", "LOF"],
        "Outlier Count": [iqr_count, z_count, if_count, lof_count],
        "Outlier %": [iqr_percent, z_percent, if_percent, lof_percent]
    })

    print(summary)
    print("\nML Methods Overlap (IForest & LOF):")
    print(f"Common Outliers: {overlap_ml} ({overlap_percent:.2f}%)")

    # -------------------------------------------------------
    # RECOMMENDATION LOGIC
    # -------------------------------------------------------
    print("\n==============================")
    print(" RECOMMENDATION ")
    print("==============================")

    if iqr_percent < 3 and z_percent < 3:
        print("✔ Classical methods show LOW outliers.")
        print("✔ Dataset is stable.")
    else:
        print("⚠ Moderate classical outliers detected.")

    if overlap_percent < 2:
        print("✔ ML methods do NOT strongly agree.")
        print("✔ Likely natural survey variation.")
    else:
        print("⚠ Strong ML agreement on anomalies.")

    print("\nOutlier analysis completed.\n")

    return summary