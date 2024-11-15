from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import numpy as np

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    # Flatten the predictions and labels if needed
    predictions = predictions.squeeze()
    labels = labels.squeeze()

    # Mean Squared Error (MSE)
    mse = mean_squared_error(labels, predictions)

    # Mean Absolute Error (MAE)
    mae = mean_absolute_error(labels, predictions)

    # Pearson Correlation
    pearson_corr, _ = pearsonr(predictions, labels)

    # Spearman Rank Correlation
    spearman_corr, _ = spearmanr(predictions, labels)

    # R-squared (Coefficient of Determination)
    r2 = r2_score(labels, predictions)

    return {
        "mse": mse,
        "mae": mae,
        "pearson_corr": pearson_corr,
        "spearman_corr": spearman_corr,
        "r2": r2
    }