# src/eda.py

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (8, 5)


# -------------------------------------
# SAFE FILENAME FUNCTION
# -------------------------------------

def clean_filename(name):
    """
    Remove special characters from column names
    to make them safe for saving as file names.
    """
    name = re.sub(r"[^\w\s-]", "", name)
    name = name.replace(" ", "_")
    return name


# -------------------------------------
# MAIN EDA FUNCTION
# -------------------------------------

def run_eda(df):

    print("\n==============================")
    print(" STARTING EDA")
    print("==============================")

    # -------------------------------------
    # Create Output Directories
    # -------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    outputs_dir = os.path.join(base_dir, "outputs")
    figures_dir = os.path.join(outputs_dir, "figures")
    tables_dir = os.path.join(outputs_dir, "tables")

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # -------------------------------------
    # 1️⃣ TARGET ANALYSIS
    # -------------------------------------

    target_distribution = df["Depression"].value_counts(normalize=True) * 100
    target_distribution.to_csv(
        os.path.join(tables_dir, "target_distribution.csv")
    )

    plt.figure()
    sns.countplot(x="Depression", data=df)
    plt.title("Depression Class Distribution")
    plt.savefig(os.path.join(figures_dir, "target_distribution.png"))
    plt.close()

    # -------------------------------------
    # 2️⃣ NUMERICAL ANALYSIS
    # -------------------------------------

    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

    if "Depression" in numerical_cols:
        numerical_cols.remove("Depression")

    # Summary statistics
    summary_stats = df[numerical_cols].describe()
    summary_stats.to_csv(
        os.path.join(tables_dir, "numerical_summary_statistics.csv")
    )

    # Correlation matrix
    corr_matrix = df.corr(numeric_only=True)
    corr_matrix.to_csv(
        os.path.join(tables_dir, "correlation_matrix.csv")
    )

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(figures_dir, "correlation_heatmap.png"))
    plt.close()

    # Boxplots vs Depression
    for col in numerical_cols:

        safe_col = clean_filename(col)

        plt.figure()
        sns.boxplot(x="Depression", y=col, data=df)
        plt.title(f"{col} vs Depression")
        plt.savefig(
            os.path.join(figures_dir, f"{safe_col}_vs_depression.png")
        )
        plt.close()

    # -------------------------------------
    # 3️⃣ CATEGORICAL ANALYSIS
    # -------------------------------------

    categorical_cols = df.select_dtypes(include="object").columns.tolist()

    for col in categorical_cols:

        safe_col = clean_filename(col)

        # Save value counts
        value_counts = df[col].value_counts()
        value_counts.to_csv(
            os.path.join(tables_dir, f"{safe_col}_value_counts.csv")
        )

        # Plot
        plt.figure(figsize=(8, 4))
        sns.countplot(x=col, hue="Depression", data=df)
        plt.title(f"{col} vs Depression")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(
            os.path.join(figures_dir, f"{safe_col}_vs_depression.png")
        )
        plt.close()

    # -------------------------------------
    # 4️⃣ MULTICOLLINEARITY CHECK
    # -------------------------------------

    corr_abs = df[numerical_cols].corr().abs()

    high_corr_pairs = []

    for i in range(len(corr_abs.columns)):
        for j in range(i):
            if corr_abs.iloc[i, j] > 0.8:
                high_corr_pairs.append(
                    (
                        corr_abs.columns[i],
                        corr_abs.columns[j],
                        corr_abs.iloc[i, j],
                    )
                )

    high_corr_df = pd.DataFrame(
        high_corr_pairs,
        columns=["Feature 1", "Feature 2", "Correlation"],
    )

    high_corr_df.to_csv(
        os.path.join(tables_dir, "high_multicollinearity_pairs.csv"),
        index=False,
    )

    print("EDA outputs successfully saved in 'outputs/' folder.")

