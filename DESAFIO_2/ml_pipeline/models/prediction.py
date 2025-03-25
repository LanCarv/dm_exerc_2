# ml_pipeline/models/prediction.py
import pandas as pd
import numpy as np

def gerar_previsoes_com_melhor_modelo(df_test, df_features, modelo_avaliado, features_usadas, nome_arquivo='previsoes.csv'):
    df_test = df_test.copy()
    df_test['date_block_num'] = 34  # mês de previsão

    df_merged = df_test.merge(
        df_features[df_features['date_block_num'] == 34],
        on=['shop_id', 'item_id', 'date_block_num'],
        how='left'
    )

    X_test = df_merged[features_usadas].fillna(0)
    y_pred = modelo_avaliado.predict(X_test)
    y_pred = np.clip(y_pred, 0, 20) # Premissa do desafio

    df_submissao = pd.DataFrame({
        'ID': df_test['ID'],
        'item_cnt_month': y_pred
    })

    df_submissao.to_csv(nome_arquivo, index=False)
    print(f"Arquivo '{nome_arquivo}' gerado com sucesso com o melhor modelo!")

    return df_submissao
