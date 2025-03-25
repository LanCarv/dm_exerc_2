from ml_pipeline.etl import importar_dados, extrair_zip_automaticamente
from ml_pipeline.features.feature_engineering import *
from ml_pipeline.models.training import avaliar_modelos
from ml_pipeline.models.prediction import gerar_previsoes_com_melhor_modelo
from ml_pipeline.utils.io import salvar_csv
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import mlflow

print("\n Iniciando pipeline de previsão de vendas...")

# 1. Extrair arquivos ZIP e carregar dados
extrair_zip_automaticamente()
X_train, test, items, item_categories, shops = importar_dados("./dados/arquivos")

# 2. Engenharia de features
X_train = criar_coluna_ano_mes(X_train)
df_mes = agrupar_vendas_mensal(X_train)
df_mes = create_lag_features(df_mes, lags=[1, 2, 3])
df_mes.fillna(0, inplace=True)
df_mes = truncar_vendas_maximas(df_mes)
df_mes = criar_variaveis_temporais(df_mes)
df_mes = criar_variaveis_top_vendas(df_mes)

# 3. Preparar treino
features_numericas = [
    'item_cnt_month_lag_1', 'item_cnt_month_lag_2', 'item_cnt_month_lag_3',
    'month', 'year', 'is_december',
    'total_sales_per_item', 'is_top_seller',
    'total_sales_per_shop', 'total_sales_per_category'
]

df_train = df_mes[df_mes['date_block_num'] < 34].copy()
X = df_train[features_numericas].fillna(0)
y = df_train['item_cnt_month']

# 4. Treinar modelos
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

# 5. Prever
test['date_block_num'] = 34
df_submissao = gerar_previsoes_com_melhor_modelo(
    df_test=test,
    df_features=df_mes,
    modelo_avaliado=melhor_modelo,
    features_usadas=features_numericas,
    nome_arquivo="dados/previsoes/previsoes.csv"
)

salvar_csv(df_submissao, "dados/previsoes/previsoes.csv")
print("\n Pipeline concluído com sucesso!")