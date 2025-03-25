# ml_pipeline/features/feature_engineering.py
import pandas as pd
import numpy as np

def criar_coluna_ano_mes(df):
    df['date'] = pd.to_datetime(df['date'], format='%d.%m.%Y')
    df['ano_mes'] = df['date'].dt.to_period('M').astype(str)
    return df

def agrupar_vendas_mensal(df):
    df_mensal = (
        df.groupby(['date_block_num', 'shop_id', 'item_id'], as_index=False)
          .agg({'item_cnt_day': 'sum'})
          .rename(columns={'item_cnt_day': 'item_cnt_month'})
    )
    return df_mensal

def adicionar_lags(df, lags=[1,2,3]):
    for lag in lags:
        df[f'item_cnt_month_lag_{lag}'] = df.groupby(['shop_id', 'item_id'])['item_cnt_month'].shift(lag)
    return df

def create_lag_features(df, lags=[1, 2, 3], col_target='item_cnt_month'):
    df_lag = df.copy()
    for lag in lags:
        lagged = df[['shop_id', 'item_id', 'date_block_num', col_target]].copy()
        lagged['date_block_num'] += lag
        lagged = lagged.rename(columns={col_target: f'{col_target}_lag_{lag}'})
        df_lag = df_lag.merge(lagged, on=['shop_id', 'item_id', 'date_block_num'], how='left')
    return df_lag

def criar_variaveis_temporais(df):
    df['ano_mes'] = pd.to_datetime(df['date_block_num'], format='%m') + pd.DateOffset(months=0)
    df['month'] = df['ano_mes'].dt.month
    df['year'] = df['ano_mes'].dt.year
    df['is_december'] = (df['month'] == 12).astype(int)
    return df

def criar_variaveis_top_vendas(df):
    total_por_item = df.groupby('item_id')['item_cnt_month'].sum()
    top_items = total_por_item.sort_values(ascending=False).head(10).index
    df['is_top_seller'] = df['item_id'].isin(top_items).astype(int)

    total_por_loja = df.groupby('shop_id')['item_cnt_month'].sum()
    df['total_sales_per_shop'] = df['shop_id'].map(total_por_loja)

    total_por_categoria = df.groupby('item_category_id')['item_cnt_month'].sum()
    df['total_sales_per_category'] = df['item_category_id'].map(total_por_categoria)

    total_por_item_final = df.groupby('item_id')['item_cnt_month'].sum()
    df['total_sales_per_item'] = df['item_id'].map(total_por_item_final)

    return df

def truncar_vendas_maximas(df, coluna='item_cnt_month', limite_superior=20):
    df[coluna] = df[coluna].clip(0, limite_superior)
    return df
