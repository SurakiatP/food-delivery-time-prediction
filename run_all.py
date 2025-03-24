import pandas as pd
import joblib

from src.preprocessing import preprocess_dataframe
from src.pipeline import build_pipeline
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import report generators
from generate_model_performance import generate_model_performance_report
from generate_feature_insights import generate_feature_insights_report
from generate_eda_summary import generate_eda_summary_report

# Step 1: Load raw data
print("📥 Loading raw dataset...")
df_raw = pd.read_csv("data/deliverytime.csv")

# Step 2: Preprocess data
print("🔧 Preprocessing dataset...")
df_cleaned = preprocess_dataframe(df_raw)
df_cleaned.to_csv("data/deliverytime_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as deliverytime_cleaned.csv")

# Step 3: Train-test split
X = df_cleaned.drop("Time_taken(min)", axis=1)
y = df_cleaned["Time_taken(min)"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train model pipeline
print("🚀 Training model...")
model = HistGradientBoostingRegressor(random_state=42)
pipeline = build_pipeline(model)
pipeline.fit(X_train, y_train)

# Step 5: Evaluate model
y_pred = pipeline.predict(X_test)
scores = evaluate_model(y_test, y_pred)

mae, rmse, r2 = scores["MAE"], scores["RMSE"], scores["R2"]

print("\n📊 Model Evaluation:")
for metric, score in scores.items():
    print(f"{metric}: {score:.4f}")

# Step 6: Save model
joblib.dump(pipeline, "models/best_model.pkl")
print("💾 Model saved to models/best_model.pkl")

# Step 7: Generate Markdown Reports
print("📝 Generating markdown reports...")
generate_model_performance_report()
generate_feature_insights_report()
generate_eda_summary_report()
print("✅ Reports generated in reports/")
