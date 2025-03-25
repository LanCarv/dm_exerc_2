# ml_pipeline/models/training.py
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from evaluation.metrics import concordance_correlation_coefficient
import matplotlib.pyplot as plt
import seaborn as sns


def ajustar_xgboost_com_grid(X, y, cv, verbose=True):
    param_grid = {
        'n_estimators': [100, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 6],
        'subsample': [0.8, 1.0]
    }
    xgb = XGBRegressor(random_state=42)
    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring='neg_root_mean_squared_error',
        cv=cv,
        n_jobs=-1,
        verbose=0
    )
    grid.fit(X, y)
    if verbose:
        print(f"GridSearch XGBoost - Melhor RMSE: {-grid.best_score_:.4f}")
        print(f"Melhores parâmetros: {grid.best_params_}")
    return grid.best_estimator_


def avaliar_modelos(X, y, modelos, n_splits=5, verbose=True):
    resultados = []
    tscv = TimeSeriesSplit(n_splits=n_splits)

    for nome, modelo in modelos.items():
        if verbose: print(f"\n Avaliando modelo: {nome}")

        if nome == 'XGBoost':
            modelo = ajustar_xgboost_com_grid(X, y, tscv, verbose=verbose)

        y_true_all = []
        y_pred_all = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
            y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

            modelo.fit(X_train_cv, y_train_cv)
            y_pred_cv = modelo.predict(X_val_cv)

            y_true_all.extend(y_val_cv)
            y_pred_all.extend(y_pred_cv)

        y_true_all = np.array(y_true_all)
        y_pred_all = np.clip(np.array(y_pred_all), 0, 20)

        rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))
        mae = mean_absolute_error(y_true_all, y_pred_all)
        r2 = r2_score(y_true_all, y_pred_all)
        mape = np.mean(np.abs((y_true_all - y_pred_all) / (y_true_all + 1e-5))) * 100
        ccc = concordance_correlation_coefficient(y_true_all, y_pred_all)

        resultados.append({
            'Modelo': nome,
            'RMSE': rmse,
            'MAE': mae,
            'MAPE': mape,
            'R²': r2,
            'CCC': ccc
        })

        plt.figure(figsize=(5, 5))
        sns.scatterplot(x=y_true_all, y=y_pred_all, alpha=0.3)
        plt.plot([0, 20], [0, 20], color='red', linestyle='--')
        plt.title(f'Real vs Predito - {nome}')
        plt.xlabel('Valor Real')
        plt.ylabel('Predição')
        plt.grid(True)
        plt.xlim(0, 20)
        plt.ylim(0, 20)
        plt.tight_layout()
        plt.show()

    df_resultados = pd.DataFrame(resultados).sort_values(by='RMSE')

    if verbose:
        print("\nResumo de Desempenho dos Modelos:")
        display(df_resultados)

    nome_melhor = df_resultados.iloc[0]['Modelo']
    melhor_modelo = modelos[nome_melhor]
    print(f"\n🏆 Melhor modelo: {nome_melhor}")
    return df_resultados, nome_melhor, melhor_modelo
