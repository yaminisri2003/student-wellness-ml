import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


class EDA:
    def __init__(self, df: pd.DataFrame, output_dir="reports"):
        self.df = df
        self.output_dir = output_dir

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    # -----------------------------------
    # 1. Target Distribution
    # -----------------------------------
    def plot_target_distribution(self):
        plt.figure()
        self.df["Wellness_Level"].value_counts().plot(kind="bar")
        plt.title("Wellness Level Distribution")
        plt.xlabel("Wellness Level")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/wellness_distribution.png")
        plt.close()
        print("Target distribution plot saved")

    # -----------------------------------
    # 2. Correlation Heatmap
    # -----------------------------------
    def plot_correlation_heatmap(self):
        plt.figure()
        numerical_df = self.df.select_dtypes(include=["int64", "float64"])
        correlation = numerical_df.corr()
        sns.heatmap(correlation, annot=False)
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/correlation_heatmap.png")
        plt.close()
        print("Correlation heatmap saved")

    # -----------------------------------
    # 3. Stress vs Wellness
    # -----------------------------------
    def plot_stress_vs_wellness(self):
        plt.figure()
        sns.boxplot(x="Wellness_Level", y="Stress_Level", data=self.df)
        plt.title("Stress Level vs Wellness")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/stress_vs_wellness.png")
        plt.close()
        print("Stress vs Wellness plot saved")

    # -----------------------------------
    # 4. Anxiety vs Wellness
    # -----------------------------------
    def plot_anxiety_vs_wellness(self):
        plt.figure()
        sns.boxplot(x="Wellness_Level", y="Anxiety_Score", data=self.df)
        plt.title("Anxiety Score vs Wellness")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/anxiety_vs_wellness.png")
        plt.close()
        print("Anxiety vs Wellness plot saved")

    # -----------------------------------
    # 5. CGPA vs Wellness
    # -----------------------------------
    def plot_cgpa_vs_wellness(self):
        plt.figure()
        sns.boxplot(x="Wellness_Level", y="CGPA", data=self.df)
        plt.title("CGPA vs Wellness Level")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/cgpa_vs_wellness.png")
        plt.close()
        print("CGPA vs Wellness plot saved")

    # -----------------------------------
    # Run All EDA
    # -----------------------------------
    def run_eda(self):
        print("Running EDA...")
        self.plot_target_distribution()
        self.plot_correlation_heatmap()
        self.plot_stress_vs_wellness()
        self.plot_anxiety_vs_wellness()
        self.plot_cgpa_vs_wellness()
        print("EDA completed")

