# 🍽️ Food Delivery Time Prediction Project

Predicting food delivery time using machine learning techniques, including feature engineering, regression modeling, and model evaluation.

---

## 🧱 Project Workflow Architecture

![Workflow Architecture](![alt text](Project_architecture.png))

---

## ✅ Goals

- Predict delivery time (in minutes) based on order, location, and delivery person information.
- Improve accuracy and reliability of delivery time predictions.
- Explore feature relationships and design meaningful engineered features.
- Build and deploy a reproducible ML pipeline for efficient model training and evaluation.

---

## 📊 Dataset

- Source: [Kaggle - Food Delivery Time Dataset](https://www.kaggle.com/datasets/rajatkumar30/food-delivery-time)
- Format: CSV (`deliverytime.csv`)
- Target: `Time_taken(min)`
- Features include:
  - Restaurant & delivery coordinates
  - Order type (Snack, Meal, etc.)
  - Delivery vehicle
  - Delivery person age and rating

---

## 💡 Selected ML-Solvable Problem

> "Can we accurately predict delivery time for a food order based on known order and delivery metadata?"

- Type: Regression Problem
- Motivation: Improve user experience, optimize logistics, and reduce delays.

---

## 📂 Project Structure

```
food-delivery-time-prediction/
├── data/
│   ├── deliverytime.csv
│   └── deliverytime_cleaned.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation.ipynb
│   └── 05_test_model_by_unseen_data.ipynb
├── reports/
│   ├── images
│   │   ├── eda
│   │   ├── features
│   │   └── model
│   ├── eda_summary.md
│   ├── feature_insights.md
│   ├── model_performance.md
│   └── improvements_and_next_steps.md
├── src/
│   ├── distance_calc.py
│   ├── evaluate.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── train_model.py
│   └── utils.py
├── models/
│   └── best_model.pkl
├── run_all.py
├── generate_eda_summary.py
├── generate_feature_insights.py
└── generate_model_performance.py
```

---

## 📐 ETL and EDA

- **Extract**: Loaded from CSV via pandas
- **Transform**:
  - Dropped unnecessary columns
  - Calculated distance using lat/lon with geopy
  - Encoded categorical variables
- **Load**: Saved cleaned CSV as `deliverytime_cleaned.csv`

EDA insights:
- Ratings and age are positively correlated with delivery time.
- Order type and vehicle type affect delivery time significantly.
- Many real-world delivery patterns are non-linear.

---

## 🪐 Feature Engineering

- Created:
  - `log_distance`: log-transformed distance
  - `age_x_rating`: interaction between age and rating
  - `speed_estimate`: estimated delivery speed (km/min)
- Applied:
  - Label Encoding
  - PCA (optional)
  - Feature importance ranking

---

## 🚀 Machine Learning Workflow

- Models tested:
  - Linear Regression
  - Random Forest
  - XGBoost
  - HistGradientBoosting
  - Stacking Regressor
- Evaluated using:
  - MAE, RMSE, R²
- Best model: **HistGradientBoostingRegressor**
  - R² Score: **0.99**
  - Cross-validated R²: **0.83**

---

## ⚙️ Pipeline

Built using `scikit-learn.Pipeline`:
- Preprocessing + Feature Engineering + Model = Single object
- Modular and maintainable (`src/pipeline.py`, `src/preprocessing.py`, etc.)

Run `run_all.py` to execute the full flow:
```bash
python run_all.py
```

---

## 🔍 Key Findings

- **Ratings** and **log_distance** are top features.
- Non-linear models (e.g., HistGB, Stacking) outperform linear models.
- Feature engineering significantly boosts model performance.

---

## 🚀 Conclusion and Next Steps

### What we learned:
- Feature interactions and transformation matter greatly.
- Model performance improves drastically with stacking and engineered features.

### What could be done differently:
- Try SHAP/LIME for explainability
- Deploy model via API (Flask/FastAPI)
- Integrate with real-time data (GPS, traffic)

### Future Work:
- Live inference dashboard
- Model monitoring over time
- More robust outlier handling

---

## 📅 Acknowledgements

- Dataset: [Kaggle - Food Delivery Time](https://www.kaggle.com/datasets/rajatkumar30/food-delivery-time)
- Inspiration: University of Chicago’s Data Project Scoping Guide

---

## Setup

### How to Run this Project

1. Clone this repository.

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Run all from the root folder:

```bash
python run_all.py
```

The results, including models, reports, and visualizations, will be saved in their respective folders.

---

## Project Author

| Name           | Contact Information                                                  |
|----------------|----------------------------------------------------------------------|
| **Surakiat P.** |                                                                      |
| 📧 Email       | [surakiat.0723@gmail.com](mailto:surakiat.0723@gmail.com)   |
| 🔗 LinkedIn    | [linkedin.com/in/surakiat](https://www.linkedin.com/in/surakiat-kansa-ard-171942351/)     |
| 🌐 GitHub      | [github.com/SurakiatP](https://github.com/SurakiatP)                 |