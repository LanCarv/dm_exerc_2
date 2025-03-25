# api_pipeline.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import shutil
import os

from ml_pipeline.etl import importar_dados
from ml_pipeline.features.feature_engineering import *
from ml_pipeline.models.training import avaliar_modelos
from ml_pipeline.models.prediction import gerar_previsoes_com_melhor_modelo
from ml_pipeline.utils.io import salvar_csv
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import mlflow

app = FastAPI(title="API - Previsão de Vendas Mensais")

X_train, test, items, item_categories, shops = None, None, None, None, None
df_mes_lags, melhor_modelo, features_numericas = None, None, None

@app.get("/")
def read_root():
    return {"mensagem": "API ativa com sucesso!"}

@app.post("/etl")
def executar_etl():
    global X_train, test, items, item_categories, shops
    try:
        X_train, test, items, item_categories, shops = importar_dados("./data/raw")
        return {"status": "ETL finalizado com sucesso."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})

@app.post("/features")
def executar_feature_engineering():
    global df_mes_lags, X_train, features_numericas
    try:
        X_train = criar_coluna_ano_mes(X_train)
        df_mes = agrupar_vendas_mensal(X_train)
        df_mes_lags = create_lag_features(df_mes, lags=[1, 2, 3])
        df_mes_lags.fillna(0, inplace=True)
        df_mes_lags = truncar_vendas_maximas(df_mes_lags)
        df_mes_lags = criar_variaveis_temporais(df_mes_lags)
        df_mes_lags = criar_variaveis_top_vendas(df_mes_lags)

        features_numericas = [
            'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3',
            'month', 'year', 'is_december',
            'total_sales_per_item', 'is_top_seller',
            'total_sales_per_shop', 'total_sales_per_category'
        ]
        return {"status": "Features criadas com sucesso."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})

@app.post("/treinar")
def treinar_modelos():
    global melhor_modelo
    try:
        df_train = df_mes_lags[df_mes_lags['date_block_num'] < 34].copy()
        X = df_train[features_numericas].fillna(0)
        y = df_train['item_cnt_month']

        modelos = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        }

        with mlflow.start_run():
            resultados, nome_melhor, melhor_modelo = avaliar_modelos(X, y, modelos)
            melhor_modelo.fit(X, y)
            mlflow.log_params(melhor_modelo.get_params())
            mlflow.log_metric("rmse", resultados.iloc[0]['RMSE'])

        return {"status": f"Modelo '{nome_melhor}' treinado com sucesso."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})

@app.post("/prever")
def prever():
    try:
        test['date_block_num'] = 34
        df_resultado = gerar_previsoes_com_melhor_modelo(
            df_test=test,
            df_features=df_mes_lags,
            modelo_avaliado=melhor_modelo,
            features_usadas=features_numericas,
            nome_arquivo='data/previsoes/previsoes.csv'
        )
        salvar_csv(df_resultado, 'dados/previsoes/previsoes.csv')
        return FileResponse(path='dados/previsoes/previsoes.csv', filename='previsoes.csv', media_type='text/csv')
    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})

@app.post("/pipeline_completo")
def rodar_tudo():
    try:
        executar_etl()
        executar_feature_engineering()
        treinar_modelos()
        return prever()
    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})