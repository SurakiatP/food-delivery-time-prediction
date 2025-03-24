# generate_eda_summary.py

import pandas as pd
import os

def generate_eda_summary_report():
    # Load raw data for EDA
    df = pd.read_csv("data/deliverytime.csv")

    # Prepare basic summaries
    num_rows = df.shape[0]
    num_columns = df.shape[1]
    missing = df.isnull().sum()
    categorical = df.select_dtypes(include='object').columns.tolist()
    numerical = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # Save summary to markdown
    os.makedirs("reports", exist_ok=True)
    with open("reports/eda_summary.md", "w", encoding="utf-8") as f:
        f.write("# 📊 EDA Summary\n\n")
        f.write(f"Dataset contains **{num_rows}** rows and **{num_columns}** columns.\n\n")

        f.write("## 🔍 Missing Values\n")
        f.write("\n```text\n")
        f.write(f"{missing.to_string()}\n")
        f.write("```\n\n")

        f.write("## 🔢 Numerical Features\n")
        f.write(", ".join(numerical) + "\n\n")

        f.write("## 🔠 Categorical Features\n")
        f.write(", ".join(categorical) + "\n\n")

        f.write("## 📝 Notes\n")
        f.write("- Delivery time (`Time_taken(min)`) appears to be continuous, suitable for regression.\n")
        f.write("- Distance can be calculated from lat/long.\n")
        f.write("- `Type_of_order` and `Type_of_vehicle` are categorical and require encoding.\n")
        f.write("- Ratings and age may have skewed distribution, which needs checking in EDA notebook.\n")

    print("📄 eda_summary.md has been generated in the reports folder.")

if __name__ == "__main__":
    generate_eda_summary()
