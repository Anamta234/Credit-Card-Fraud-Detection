<div align="center">

# 🛡️ Credit Card Fraud Detection

### End-to-end Machine Learning pipeline for detecting fraudulent transactions

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data-150458?style=for-the-badge&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-2ea44f?style=for-the-badge)

<br/>

> Analyzed **339,607** real-world credit card transactions · Detected fraud patterns across time, location & category · Achieved **85%+ accuracy** with Random Forest

</div>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Project Workflow](#-project-workflow)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Feature Engineering](#-feature-engineering)
- [Model](#-model)
- [Results](#-results)
- [Key Insights](#-key-insights)
- [Quick Start](#-quick-start)
- [Tech Stack](#-tech-stack)

---

## 🔍 Overview

Credit card fraud causes billions in losses annually. This project builds a complete **fraud detection pipeline** — from raw messy data to a deployed prediction model — using classical machine learning and rigorous EDA.

```
Raw Transactions → Cleaning → EDA → Feature Engineering → Model → Prediction
```

**What makes this project stand out:**
- Tackles a severely **imbalanced dataset** (fraud < 1% of data)
- Engineers **4 custom features** from timestamps and geolocation
- Uses an end-to-end **Scikit-learn Pipeline** (no data leakage)
- Visualizes fraud patterns across time, geography, and spending category

---

## 📦 Dataset

| Property | Value |
|---|---|
| File | `credit_card_frauds.parquet` |
| Rows | 339,607 transactions |
| Columns | 15 features |
| Fraud rate | ~0.52% (highly imbalanced) |
| Time span | 2019 – 2020 |

<details>
<summary><b>📋 Column Descriptions (click to expand)</b></summary>

<br/>

| Column | Type | Description |
|---|---|---|
| `trans_date_trans_time` | datetime | Timestamp of the transaction |
| `merchant` | string | Store or merchant name |
| `category` | string | Purchase category (grocery, shopping, etc.) |
| `amt` | float | Transaction amount in USD |
| `city` | string | Customer's city |
| `state` | string | Customer's state |
| `lat` / `long` | float | Customer's GPS coordinates |
| `city_pop` | int | Population of customer's city |
| `job` | string | Customer's occupation |
| `dob` | string | Customer's date of birth |
| `trans_num` | string | Unique transaction identifier |
| `merch_lat` / `merch_long` | float | Merchant's GPS coordinates |
| `is_fraud` | int | **Target** — 0 = Legitimate, 1 = Fraud |

</details>

---

## 🔄 Project Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PROJECT PIPELINE                             │
├──────────┬────────────┬──────────────┬─────────────┬───────────────┤
│  Load &  │  Handle    │     EDA      │  Feature    │  Train &      │
│  Inspect │ Imbalance  │  (10 plots)  │ Engineering │  Evaluate     │
│          │            │              │             │               │
│ Parquet  │Undersample │ Time/Geo/Cat │ age distance│ RandomForest  │
│ 339K rows│ → 4K rows  │  analysis    │ hour  day   │ 85%+ accuracy │
└──────────┴────────────┴──────────────┴─────────────┴───────────────┘
```

---

## 📊 Exploratory Data Analysis

Ten visualizations reveal where, when, and how fraud happens:

### 🕐 Time Patterns

| Dimension | Finding |
|---|---|
| **Hour of day** | Fraud peaks at **22:00–23:00** (459 & 452 cases) |
| **Month** | **March** (220 cases) and **September** (197 cases) are highest |
| **Year** | Slight decline — 978 cases in 2019 → 804 in 2020 |

### 📍 Geographic Patterns

| Dimension | Finding |
|---|---|
| **State** | **California** leads (402 cases), followed by Missouri (262) |
| **City** | **Albuquerque** (24), Aurora (23), Fort Washakle highest |
| **Population** | Larger cities tend to have more fraud cases |

### 🛒 Behavioral Patterns

| Dimension | Finding |
|---|---|
| **Category** | **grocery_pos** and **shopping_net** most targeted |
| **Amount** | Most transactions are $0–200; high-value = suspicious |
| **Correlation** | `amt` vs `is_fraud` = **0.64** — moderate positive signal |

---

## ⚙️ Feature Engineering

Four new features were derived to boost model performance:

```python
# Age from date of birth
df['age'] = 2026 - pd.to_datetime(df['dob']).dt.year

# Euclidean distance between customer & merchant
df['distance'] = np.sqrt((df['lat'] - df['merch_lat'])**2 +
                         (df['long'] - df['merch_long'])**2)

# Time-based features from transaction timestamp
df['hours'] = pd.to_datetime(df['trans_date_trans_time']).dt.hour
df['day']   = pd.to_datetime(df['trans_date_trans_time']).dt.dayofweek
```

> Distance captures geographic anomalies — fraudulent transactions often occur far from the customer's usual location.

---

## 🤖 Model

### Handling Class Imbalance

The original dataset had **< 1% fraud** — a severe imbalance that would cause models to ignore fraud entirely.

**Solution:** Undersampling the majority class

```
Before:  337,389 legitimate  +  2,218 fraud   (ratio 152:1)
After:     2,218 legitimate  +  1,782 fraud   (ratio ~1.2:1)
```

### Pipeline Architecture

```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

pipe = Pipeline([
    ('encoder', ColumnTransformer([
        ('ohe', OneHotEncoder(handle_unknown='ignore'), ['category', 'state'])
    ], remainder='passthrough')),

    ('model', RandomForestClassifier(
        n_estimators = 300,
        max_depth    = 6,
        min_samples_split = 5,
        random_state = 42
    ))
])
```

**Why Random Forest?**
- Handles mixed data types natively
- Robust to outliers (which exist in transaction amounts)
- No feature scaling required
- Resistant to overfitting via ensemble averaging

---

## 📈 Results

| Metric | Score |
|---|---|
| Test Accuracy | **85%+** |
| Cross-validation (5-fold) | Stable across all folds |
| Key predictor | Transaction amount (`amt`) |

### Sample Prediction

```python
sample = pd.DataFrame([{
    'category': 'mis',
    'amt': 321,
    'state': 'AZ',
    'age': 56,
    'distance': 0,
    'hours': 0,
    'day': 8
}])

pipe.predict(sample)
# → "The transaction is predicted to be fraudulent."
```

---

## 💡 Key Insights

1. **Time is a strong signal** — fraudsters operate at night (10PM–midnight) when monitoring is lower
2. **Geography matters** — California and a handful of cities account for a disproportionate share of fraud
3. **No single feature is enough** — amount alone has 0.64 correlation; multi-feature modeling is essential
4. **Certain categories are honeypots** — grocery POS and online shopping need extra scrutiny
5. **Fraud is declining** — slight drop from 2019 to 2020 suggests improving detection systems

---

## 🚀 Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/your-username/credit-card-fraud-detection.git
cd credit-card-fraud-detection

# 2. Install dependencies
pip install pandas numpy seaborn matplotlib scikit-learn pyarrow jupyter

# 3. Launch the notebook
jupyter notebook itds_project.ipynb
```

---

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Numerical operations & feature engineering |
| `matplotlib` / `seaborn` | EDA visualizations (10 plots) |
| `scikit-learn` | Pipeline, encoding, Random Forest, evaluation |
| `pyarrow` | Parquet file format support |

---

## 📁 Repository Structure

```
credit-card-fraud-detection/
│
├── 📓 itds_project.ipynb        # Full notebook — EDA + modeling
├── 📊 credit_card_frauds.parquet # Raw transaction dataset
└── 📄 README.md
```

---

<div align="center">

Made with ❤️ for Introduction to Data Science · 2026

</div>
