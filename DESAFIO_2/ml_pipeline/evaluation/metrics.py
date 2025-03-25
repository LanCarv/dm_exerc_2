# ml_pipeline/evaluation/metrics.py
import numpy as np

def concordance_correlation_coefficient(y_true, y_pred):
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred) ** 2 + 1e-5)
    return ccc

def calcular_metricas(y_true, y_pred):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    y_pred = np.clip(y_pred, 0, 20)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-5))) * 100
    ccc = concordance_correlation_coefficient(y_true, y_pred)

    return {
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R²': r2,
        'CCC': ccc
    }