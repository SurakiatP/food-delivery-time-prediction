# 🧠 Model Architecture

## 🏠 Pipeline Structure
The machine learning pipeline is built using `scikit-learn`'s `Pipeline` module.
It integrates preprocessing and modeling into a single streamlined process.

### Steps:
1. **Preprocessing**
   - Custom feature engineering from `preprocessing.py`
   - Engineered features: `log_distance`, `age_x_rating`, `speed_estimate`, `distance_x_rating`
   - Label Encoding and One-Hot Encoding for categorical variables
2. **StandardScaler**
   - Applied only to models that require feature scaling
3. **Regressor**
   - Final model: `HistGradientBoostingRegressor`

---

## 🧠 Models Evaluated
- Linear Regression
- Random Forest
- XGBoost
- HistGradientBoostingRegressor ✅
- StackingRegressor (ensemble of above)

After comparing all models, `HistGradientBoostingRegressor` was selected due to its high accuracy (R² ≈ 0.99) and robustness.

---

## 🔧 Final Parameters
Model: `HistGradientBoostingRegressor(random_state=42)`
- Handles missing values internally
- No feature scaling required
- Robust with tabular data and small feature sets

---

## 🧪 Evaluation Methodology
- **Train-Test Split**: `train_test_split(test_size=0.2, random_state=42)`
- **Metrics Used**:
  - Mean Absolute Error (MAE)
  - Root Mean Squared Error (RMSE)
  - R² Score
- **Cross-validation**: 5-fold used with GridSearchCV during tuning phase

---

> ✅ The model pipeline is modular, reproducible, and suitable for real-world deployment or API integration.

