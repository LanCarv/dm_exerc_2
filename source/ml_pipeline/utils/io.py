# ml_pipeline/utils/io.py
import pickle
import pandas as pd
import os

def save_pickle(obj, path):
    """Salva um objeto Python em formato pickle."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f" Objeto salvo em: {path}")

def load_pickle(path):
    """Carrega um objeto em formato pickle."""
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    print(f" Objeto carregado de: {path}")
    return obj

def salvar_csv(df: pd.DataFrame, path: str):
    """Salva um DataFrame em CSV."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f" CSV salvo em: {path}")

def carregar_csv(path: str):
    """Carrega um DataFrame de um CSV."""
    df = pd.read_csv(path)
    print(f" CSV carregado de: {path}")
    return df
