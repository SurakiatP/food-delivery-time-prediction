import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Evaluate regression performance
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    return {'MAE': mae, 'RMSE': rmse, 'R2': r2}

# Plot actual vs predicted
def plot_actual_vs_predicted(y_true, y_pred, model_name="Model"):
    plt.figure(figsize=(8,5))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual Time Taken (min)')
    plt.ylabel('Predicted Time Taken (min)')
    plt.title(f'Actual vs Predicted - {model_name}')
    plt.tight_layout()
    plt.show()

# Plot feature importance if model supports it
def plot_feature_importance(model, feature_names):
    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        importances = model.named_steps['regressor'].feature_importances_
        if len(importances) != len(feature_names):
            print("Feature importance length mismatch. Skipping plot.")
            return
        fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        fi_df = fi_df.sort_values(by='Importance', ascending=False)

        plt.figure(figsize=(10,6))
        sns.barplot(x='Importance', y='Feature', data=fi_df)
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.show()
    else:
        print("This model does not support feature importances.")
