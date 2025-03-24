from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import HistGradientBoostingRegressor, StackingRegressor

# Build a pipeline given a model
def build_pipeline(model):
    if isinstance(model, HistGradientBoostingRegressor):
        # HistGB handles missing values and does not require scaling
        return Pipeline([
            ('regressor', model)
        ])
    else:
        return Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])

# Create a stacking model pipeline (meta-model: Linear Regression)
def build_stacking_pipeline():
    base_models = [
        ('xgb', XGBRegressor(random_state=42)),
        ('rf', RandomForestRegressor(random_state=42)),
        ('hist', HistGradientBoostingRegressor(random_state=42))
    ]
    final_estimator = LinearRegression()

    stacking = StackingRegressor(
        estimators=base_models,
        final_estimator=final_estimator,
        passthrough=True,
        n_jobs=-1
    )

    return Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', stacking)
    ])
