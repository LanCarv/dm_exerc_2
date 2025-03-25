# ml_pipeline/etl.py
import pandas as pd
import os
import zipfile
from dotenv import load_dotenv

load_dotenv()

def extrair_zip_automaticamente():
    zip_path = os.getenv("ZIP_PATH_2")
    extract_path = os.getenv("EXTRACT_PATH_2")

    if not zip_path or not os.path.exists(zip_path):
        raise FileNotFoundError(f"Arquivo ZIP não encontrado em: {zip_path}")
    if not extract_path or not os.path.exists(extract_path):
        raise FileNotFoundError(f"Pasta de destino não encontrada: {extract_path}")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
        print(f"📦 Arquivos extraídos com sucesso em: {extract_path}")

def importar_dados(caminho_base):
    X_train = pd.read_csv(f"{caminho_base}/sales_train.csv")
    y_test = pd.read_csv(f"{caminho_base}/test.csv")
    df_items = pd.read_csv(f"{caminho_base}/items.csv")
    df_item_categories = pd.read_csv(f"{caminho_base}/item_categories.csv")
    df_shops = pd.read_csv(f"{caminho_base}/shops.csv")
    return X_train, y_test, df_items, df_item_categories, df_shops

def analise_por_coluna(df: pd.DataFrame):
    analise_geral = []
    for col in df.columns:
        col_data = {
            'Coluna': col,
            'Registros': df[col].count(),
            'Nulos': df[col].isnull().sum(),
            'Perc Nulos': (df[col].isnull().sum() / df.shape[0]) * 100,
            'Registro únicos': df[col].nunique(),
            'Valor mais frequente': df[col].mode()[0] if not df[col].mode().empty else None,
            'Frequência do valor mais comum': df[col].value_counts().max() if not df[col].value_counts().empty else None,
            'Tipo dado': df[col].dtype
        }
        analise_geral.append(col_data)
    return pd.DataFrame(analise_geral)