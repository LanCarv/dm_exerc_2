# api_pipeline.py
import traceback
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
import pandas as pd
import shutil
import os

from source.ml_pipeline.etl.etl import importar_dados
from source.ml_pipeline.evaluation import metrics
from source.ml_pipeline.features.feature_engineering import *
from source.ml_pipeline.models.training import avaliar_modelos
from source.ml_pipeline.models.prediction import gerar_previsoes_com_melhor_modelo
from source.ml_pipeline.utils.io import salvar_csv
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

@app.get("/debug/colunas_df_mes_lags")
def ver_colunas_df_mes_lags():
    try:
        if df_mes_lags is None:
            return JSONResponse(status_code=404, content={"erro": "df_mes_lags ainda não foi inicializado."})
        
        colunas = df_mes_lags.columns.tolist()
        return {"colunas_df_mes_lags": colunas}
    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})

@app.post("/etl")
def executar_etl():
    global X_train, test, items, item_categories, shops
    try:
        X_train, test, items, item_categories, shops = importar_dados("./source/dados/arquivos")
        return {"status": "ETL finalizado com sucesso."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})

@app.post("/features")
def executar_feature_engineering():
    global df_mes_lags, X_train, features_numericas
    try:
        X_train = criar_coluna_ano_mes(X_train)
        df_mes = agrupar_vendas_mensal(X_train)
        df_mes = df_mes.merge(items[['item_id', 'item_category_id']], on='item_id', how='left')  # ⬅️ Adiciona categorias
        df_mes = criar_variaveis_top_vendas(df_mes)  # ⬅️ Aplica antes dos lags
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
        print("Iniciando treinamento dos modelos...")

        df_train = df_mes_lags[df_mes_lags['date_block_num'] < 34].copy()
        X = df_train[features_numericas].fillna(0)
        y = df_train['item_cnt_month']

        modelos = {
            'LinearRegression': LinearRegression(),
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
            'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
        }

        mlflow.set_tracking_uri("mlruns")
        mlflow.set_experiment("experimento_previsao_vendas")

        with mlflow.start_run():
            resultados, nome_melhor, melhor_modelo = avaliar_modelos(X, y, modelos)
            melhor_modelo.fit(X, y)
            mlflow.log_params(melhor_modelo.get_params())
            mlflow.log_metric("rmse", resultados.iloc[0]['RMSE'])

        print(f"Modelo '{nome_melhor}' treinado com sucesso.")
        return {"status": f"Modelo '{nome_melhor}' treinado com sucesso."}

    except Exception:
        import traceback
        erro = traceback.format_exc()
        print("Erro no endpoint /treinar:\n", erro)
        return JSONResponse(status_code=500, content={"erro": erro})

@app.post("/prever")
def prever():
    try:
        print("Iniciando função prever()")
        test['date_block_num'] = 34
        caminho_csv = 'source/dados/previsoes/previsoes.csv'

        df_resultado = gerar_previsoes_com_melhor_modelo(
            df_test=test,
            df_features=df_mes_lags,
            modelo_avaliado=melhor_modelo,
            features_usadas=features_numericas,
            nome_arquivo=caminho_csv
        )

        print(f"CSV foi gerado? {os.path.exists(caminho_csv)}")

        if os.path.exists(caminho_csv):
            return FileResponse(
                path=caminho_csv,
                filename="previsoes.csv",
                media_type="text/csv"
            )
        else:
            return JSONResponse(status_code=500, content={"erro": "Arquivo não encontrado após prever."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"erro": str(e)})

@app.post("/pipeline_completo")
def pipeline_completo():
    try:
        print("Iniciando execução do pipeline completo via API...")

        # ETL
        executar_etl()

        # Engenharia de Features
        executar_feature_engineering()

        # Definir features
        global features_numericas
        features_numericas = [
            'item_cnt_month_lag_1',
            'item_cnt_month_lag_2',
            'item_cnt_month_lag_3',
            'month',
            'year',
            'is_december',
            'total_sales_per_item',
            'is_top_seller',
            'total_sales_per_shop',
            'total_sales_per_category'
        ]

        # Treinar modelos
        treinar_modelos()

        # Previsão
        test['date_block_num'] = 34

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        PREVISAO_DIR = os.path.join(BASE_DIR, "source", "dados", "previsoes")
        os.makedirs(PREVISAO_DIR, exist_ok=True)
        caminho_csv = os.path.join(PREVISAO_DIR, "previsoes.csv")

        df_resultado = gerar_previsoes_com_melhor_modelo(
            df_test=test,
            df_features=df_mes_lags,
            modelo_avaliado=melhor_modelo,
            features_usadas=features_numericas,
            nome_arquivo=caminho_csv
        )

        print(f"Pipeline concluído. Arquivo salvo em: {caminho_csv}")
        return FileResponse(
            path=caminho_csv,
            filename="previsoes.csv",
            media_type="text/csv"
        )

    except Exception:
        import traceback
        erro = traceback.format_exc()
        print("ERRO DETECTADO:")
        print(erro)
        return JSONResponse(status_code=500, content={"erro": erro})
