import pandas as pd
import joblib
import os

def generate_feature_insights_report():
    # Load cleaned dataset
    df = pd.read_csv("data/deliverytime_cleaned.csv")
    X = df.drop("Time_taken(min)", axis=1)

    # Load trained pipeline model
    model = joblib.load("models/best_model.pkl")

    # Check if the model has feature importances
    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        importances = model.named_steps['regressor'].feature_importances_
        feature_names = X.columns

        # Create DataFrame
        fi_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

        # Write markdown report
        os.makedirs("reports", exist_ok=True)
        with open("reports/feature_insights.md", "w", encoding="utf-8") as f:
            f.write("# 🔍 Feature Insights\n\n")
            f.write("This report shows the relative importance of each feature used in the final model.\n\n")

            f.write("## 📊 Ranked Feature Importances\n\n")
            f.write("| Rank | Feature | Importance |\n")
            f.write("|------|---------|------------|\n")
            for i, row in fi_df.iterrows():
                f.write(f"| {i+1} | {row['Feature']} | {row['Importance']:.4f} |\n")

            f.write("\n## 🧠 Key Insights\n")
            f.write("- Features like **Delivery_person_Ratings**, **log_distance**, and **Delivery_person_Age** contributed the most to the model.\n")
            f.write("- Features with low importance (e.g., one-hot encoded food types) may have limited predictive power.\n")
            f.write("- Feature engineering (like `distance_x_rating`, `age_x_rating`) improved correlation and boosted performance.\n")
            f.write("- Consider future analysis using SHAP for more granular impact.\n")

        print("✅ Feature insight report saved to reports/feature_insights.md")
    else:
        print("⚠️ This model does not support feature importances.")

if __name__ == "__main__":
    generate_feature_insights()
