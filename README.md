#  Student Wellness ML — Depression Prediction System

A machine learning project that predicts student depression risk using behavioral, academic, and lifestyle indicators — built as a complete, end-to-end pipeline from raw data to an interactive Streamlit dashboard.

---

##  What This Project Does

Student depression is a growing concern in academic institutions. This project builds a **binary classification system** that identifies students at risk of depression using survey data. Multiple ML algorithms are trained and compared, the best model is fine-tuned using **Optuna**, and every prediction is explained using **SHAP** — making the system both accurate and interpretable.

---

##  Project Structure

```
student-wellness-ml/
│
├── app.py                        # Streamlit dashboard
├── main.py                       # Main ML pipeline runner
├── req.txt                       # Python dependencies
│
├── src/
│   ├── data_loader.py            # Step 1 — Load raw data
│   ├── data_cleaner.py           # Step 2 — Clean & prepare data
│   ├── assumption_checker.py     # Step 3 — Outlier analysis
│   ├── eda.py                    # Step 4 — Exploratory Data Analysis
│   ├── advanced_models.py       # Step 5 — Model definitions
│   ├── model_trainer.py         # Step 7 — Baseline model training
│   ├── model_evaluator.py       # Step 8 — Model comparison & CV
│   ├── hyperparameter_tuning.py # Step 8 — Optuna tuning
│   ├── feature_importance.py    # Step 9 — Permutation importance & PDP
│   └── shap_analysis.py         # Step 10 — SHAP explainability
│
├── data/
│   ├── raw/                      # Raw input dataset (not tracked)
│   └── processed/                # Cleaned datasets
│
├── outputs/
│   ├── figures/                  # EDA plots
│   └── tables/                   # EDA summary CSVs
│
├── artifacts/                    # All saved model outputs
└── .gitignore
```

---


##  End-to-End Pipeline — Step by Step

The entire project flows in **11 sequential steps**, from raw data all the way to explainable predictions. Run everything with:

```bash
python main.py
```

Then visualize results with:

```bash
streamlit run app.py
```

---

### ─────────────────────────────────────────
### STEP 1 — Load Data
### `src/data_loader.py`
### ─────────────────────────────────────────

The pipeline begins by reading the raw student survey CSV from `data/raw/`.

```python
df = load_data("data/raw/student_depression_dataset.csv")
```

- Reads the CSV using `pandas`
- Prints the dataset shape for a quick sanity check
- Returns a raw DataFrame ready for cleaning

**Output:** Raw DataFrame loaded into memory

---

### ─────────────────────────────────────────
### STEP 2 — Clean Data
### `src/data_cleaner.py`
### ─────────────────────────────────────────

The raw data is cleaned to remove noise before any analysis begins.

```python
df = clean_data(df)
```

What happens:
- Drops the `id` column (not a feature)
- Strips leading/trailing whitespace from all column names
- Drops `"Have you ever had suicidal thoughts ?"` to avoid **data leakage**
- Saves the cleaned dataset to `data/processed/`

**Output:** Clean, lean DataFrame with no leakage risk

---

### ─────────────────────────────────────────
### STEP 3 — Outlier Analysis
### `src/assumption_checker.py`
### ─────────────────────────────────────────

Before modeling, outliers are detected using **four complementary methods** to understand data quality:

| Method | How It Works |
|--------|-------------|
| **IQR** | Flags values beyond 1.5× the interquartile range |
| **Z-Score** | Flags values beyond 3 standard deviations |
| **Isolation Forest** | ML-based anomaly detection via random partitioning |
| **Local Outlier Factor (LOF)** | ML-based density anomaly detection |

The overlap between Isolation Forest and LOF is used as a reliability signal — low overlap suggests natural survey variation rather than true data errors.

A recommendation is printed automatically:
-  If classical outliers are < 3% → dataset is stable
-  If ML methods don't strongly agree → likely natural variation
-  If both flags are high → anomalies may need treatment

**Output:** Summary table printed + recommendation logged to console

---

### ─────────────────────────────────────────
### STEP 4 — Exploratory Data Analysis (EDA)
### `src/eda.py`
### ─────────────────────────────────────────

A comprehensive EDA is run to understand patterns, distributions, and relationships in the data before any modeling.

**Target Analysis**
- Class distribution plot of the `Depression` label → `outputs/figures/target_distribution.png`
- Saved as CSV → `outputs/tables/target_distribution.csv`

**Numerical Feature Analysis**
- Summary statistics (mean, std, min, max) → `outputs/tables/numerical_summary_statistics.csv`
- Correlation heatmap across all numerical features → `outputs/figures/correlation_heatmap.png`
- Boxplots of every numerical feature vs Depression label → `outputs/figures/`

**Categorical Feature Analysis**
- Value counts per category → `outputs/tables/`
- Count plots of every categorical feature vs Depression label → `outputs/figures/`

**Multicollinearity Check**
- Detects feature pairs with correlation > 0.8
- Saved to → `outputs/tables/high_multicollinearity_pairs.csv`

**Output:** Full set of EDA plots and tables in `outputs/figures/` and `outputs/tables/`

---

### ─────────────────────────────────────────
### STEP 5 — Preprocessing
### `main.py` (ColumnTransformer)
### ─────────────────────────────────────────

Features are split into numerical and categorical, then processed via a `ColumnTransformer`:

```python
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numerical_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
])
```

- **StandardScaler** normalizes numerical features to zero mean and unit variance
- **OneHotEncoder** converts categorical columns to binary dummy variables
- The train/test split is **80/20 stratified** to preserve class balance

**Output:** Preprocessed `X_train`, `X_test`, `y_train`, `y_test` arrays

---

### ─────────────────────────────────────────
### STEP 6 — Baseline Model: Logistic Regression
### `src/model_trainer.py`
### ─────────────────────────────────────────

Before comparing complex models, a **Logistic Regression** baseline is established.

**Why Logistic Regression first?**
It is simple, interpretable, and fast. It reveals whether linear relationships alone can explain depression risk, and sets a performance floor for all other models to beat.

```python
LogisticRegression(max_iter=1000, class_weight="balanced")
```

- `class_weight="balanced"` automatically handles class imbalance in the dataset
- Feature importance is extracted from absolute coefficient values
- **Top 15 most important features** are plotted → `artifacts/top_15_feature_importance.png`
- Full feature coefficients CSV → `artifacts/logistic_feature_importance.csv`

**Output:** Baseline accuracy, precision, recall, F1, classification report

---

### ─────────────────────────────────────────
### STEP 7 — Multi-Model Comparison
### `src/advanced_models.py` + `src/model_evaluator.py`
### ─────────────────────────────────────────

Four algorithms are trained and evaluated head-to-head using **5-fold Stratified Cross-Validation**, ranked by F1 Score.

#### The Four Algorithms

** Logistic Regression** *(Baseline)*
Linear classifier. Interpretable, low variance. Used to benchmark all other models.

** Random Forest**
Ensemble of decision trees trained via bagging. Reduces overfitting and handles non-linear relationships well in high-dimensional survey data.
```python
RandomForestClassifier(n_estimators=200, class_weight="balanced")
```

** Gradient Boosting**
Sequentially corrects errors of weak learners. Highly effective on tabular data and selected as the **final model** after this comparison step.
```python
GradientBoostingClassifier(n_estimators=200)
```

* XGBoost**
Optimized gradient boosting with L1/L2 regularization. Faster than standard GB and handles sparse, one-hot encoded features effectively.
```python
XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=6, eval_metric="logloss")
```

#### Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted depressed students, how many actually are |
| **Recall** | Of actually depressed students, how many were caught |
| **F1 Score** | Harmonic mean of precision & recall *(primary metric)* |
| **ROC-AUC** | Model's ability to distinguish between classes |

#### Model Comparison Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| **Gradient Boosting** *(tuned)* |  Best |  Best |  Best |  Best |  Best |
| XGBoost | High | High | High | High | High |
| Random Forest | High | High | Moderate | High | High |
| Logistic Regression | Moderate | Moderate | Moderate | Moderate | Moderate |

>  Full numeric results → `artifacts/model_comparison.csv`

**Gradient Boosting** wins on F1 and is carried forward for tuning.

---

### ─────────────────────────────────────────
### STEP 8 — Hyperparameter Tuning (Optuna)
### `src/hyperparameter_tuning.py`
### ─────────────────────────────────────────

The winning Gradient Boosting model is fine-tuned using **Optuna** — an automated hyperparameter optimization framework that uses Bayesian search to find the best configuration efficiently.

**Optimization objective:** Maximize F1 Score via 5-fold Stratified Cross-Validation

**Search space:**

| Parameter | Search Range |
|-----------|-------------|
| `n_estimators` | 100 – 1000 |
| `learning_rate` | 0.01 – 0.3 *(log scale)* |
| `max_depth` | 3 – 10 |
| `min_samples_split` | 2 – 20 |
| `min_samples_leaf` | 1 – 10 |
| `subsample` | 0.6 – 1.0 |

```python
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

The best parameters are printed and the full study is saved for reproducibility → `artifacts/gb_optuna_study.pkl`

**Output:** Optimal hyperparameters for the final Gradient Boosting model

---

### ─────────────────────────────────────────
### STEP 9 — Feature Importance & PDP
### `src/feature_importance.py`
### ─────────────────────────────────────────

After the final model is trained, two techniques reveal **which features matter most** and how they influence predictions.

**Permutation Importance**
Measures how much the model's F1 score drops when each feature's values are randomly shuffled. A large drop = the feature is important.

- Top 10 features plotted → `artifacts/permutation_importance.png`
- Full importance scores → `artifacts/permutation_importance.csv`

**Partial Dependence Plots (PDP)**
Shows the marginal effect of the top 5 features on the predicted probability of depression — holding all other features constant.

- One plot per feature → `artifacts/pdp/pdp_<feature_name>.png`

**Output:** Visual understanding of which student factors most influence depression risk

---

### ─────────────────────────────────────────
### STEP 10 — SHAP Explainability
### `src/shap_analysis.py`
### ─────────────────────────────────────────

The final step goes beyond feature importance — **SHAP (SHapley Additive exPlanations)** explains *exactly how much each feature contributed to each individual prediction*.

`shap.TreeExplainer` is used, which is optimized for tree-based models like Gradient Boosting.

```python
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_sample)
```

A sample of 2000 rows is used for computational efficiency.

**Outputs:**

| Artifact | Description | Location |
|----------|-------------|----------|
| SHAP Summary Plot (beeswarm) | Global feature impact across all predictions | `artifacts/shap/shap_summary.png` |
| SHAP Bar Plot | Mean absolute SHAP values ranked by importance | `artifacts/shap/` |
| SHAP Values CSV | Raw per-row SHAP values for all features | `artifacts/shap/shap_values.csv` |

SHAP makes the model **transparent and trustworthy** — enabling educators and counselors to understand *why* a student is flagged, not just *that* they are flagged.

---

### ─────────────────────────────────────────
### STEP 11 — Streamlit Dashboard
### `app.py`
### ─────────────────────────────────────────

All results are surfaced through an interactive web dashboard built with **Streamlit**.

```bash
streamlit run app.py
```

Dashboard sections:

| Section | What It Shows |
|---------|--------------|
| **Dataset Overview** | First 10 rows of raw data + shape |
| **Test Set Metrics** | Accuracy, Precision, Recall, F1, ROC-AUC table |
| **Confusion Matrix** | Heatmap of predicted vs actual labels |
| **Feature Importance** | Permutation importance & PDP plots |
| **SHAP Explainability** | SHAP summary and bar plots |

>  Run `python main.py` before launching the dashboard to generate all required artifacts.

---

##  Tech Stack

| Tool | Purpose |
|------|---------|
| Python | Core language |
| scikit-learn | ML models, preprocessing, evaluation, PDP |
| XGBoost | Regularized gradient boosting |
| Optuna | Automated hyperparameter optimization |
| SHAP | Model explainability (TreeExplainer) |
| Streamlit | Interactive results dashboard |
| pandas / NumPy | Data manipulation |
| matplotlib / seaborn | Visualizations |
| joblib | Study serialization |
| scipy | Z-score outlier detection |
