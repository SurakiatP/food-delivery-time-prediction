import joblib
import pandas as pd
from src.evaluate import evaluate_model
from sklearn.model_selection import train_test_split

def generate_model_performance_report(mae=None, rmse=None, r2=None):
    """
    Generates model_performance.md report using passed metrics or recomputed ones.
    """
    if mae is None or rmse is None or r2 is None:
        # Recalculate from model and test set
        df = pd.read_csv("data/deliverytime_cleaned.csv")
        model = joblib.load("models/best_model.pkl")

        X = df.drop("Time_taken(min)", axis=1)
        y = df["Time_taken(min)"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        y_pred = model.predict(X_test)
        scores = evaluate_model(y_test, y_pred)

        mae = scores["MAE"]
        rmse = scores["RMSE"]
        r2 = scores["R2"]

    with open("reports/model_performance.md", "w", encoding="utf-8") as f:
        f.write("# Model Performance\n\n")
        f.write("## Final Selected Model\n")
        f.write("**HistGradientBoostingRegressor**\n\n")
        f.write("| Metric | Value  |\n")
        f.write("|--------|--------|\n")
        f.write(f"| MAE    | {mae:.4f} |\n")
        f.write(f"| RMSE   | {rmse:.4f} |\n")
        f.write(f"| R²     | {r2:.4f} |\n\n")
        f.write("✅ HistGB was chosen due to its best overall performance.\n")

    print("📄 model_performance.md has been generated in the reports folder.")
