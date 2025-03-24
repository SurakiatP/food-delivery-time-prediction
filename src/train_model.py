import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import HistGradientBoostingRegressor

from src.preprocessing import preprocess_dataframe
from src.pipeline import build_pipeline, build_stacking_pipeline

# Load and preprocess data
df = pd.read_csv("../data/deliverytime_cleaned.csv")
df = preprocess_dataframe(df)

X = df.drop("Time_taken(min)", axis=1)
y = df["Time_taken(min)"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train model
model = HistGradientBoostingRegressor(random_state=42)
pipeline = build_pipeline(model)
pipeline.fit(X_train, y_train)

# Evaluate on test set
y_pred = pipeline.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("Model Evaluation:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

# Save model
joblib.dump(pipeline, "../models/best_model.pkl")
print("Model saved to models/best_model.pkl")
