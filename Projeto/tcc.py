import pandas as pd
from base import concat_files
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tkinter as tk
import pandas as pd
import numpy as np
from pypfopt import risk_models

#especificando os parametros utilizados para a chamada do arquivo de pre-processamento
def load_data():
    year_date = ['2018'] 
    path=f'/home/luis/Documents/b3/'
    name_file='2018'
    type_file ='TXT'
    final_file = 'all_bovespa.csv'  

    df = concat_files( path, name_file, year_date,type_file, final_file)
    return df

    # df_bradesco = df[df['sigla_acao'] == 'BBDC4' ]
    # df_banco_do_brasil = df[df['sigla_acao'] == 'BBAS3' ]
    # df_banrisul = df[df['sigla_acao'] == 'BRSR6' ]
    # df_santander = df[df['sigla_acao'] == 'SANB11' ]
    # df_inter = df[df['sigla_acao'] == 'BIDI4' ]

def calculate_returns(df_asset):
    df_asset['preco_fechamento_anterior'] = df_asset['preco_fechamento'].shift(1)
    df_asset['daily_returns'] = (df_asset['preco_fechamento'] - df_asset['preco_fechamento_anterior']) / df_asset['preco_fechamento_anterior']
    # df_asset.dropna(inplace=True)
    df_asset['daily_returns'].fillna(method='ffill', inplace=True)  # preenchendo valores nulos pelo valor mais recente
    
    return df_asset


def filtering_assets(df, sigla_acao):
    df_asset = df[df['sigla_acao'] == sigla_acao].copy()

    df_asset['data_pregao'] = pd.to_datetime(df_asset['data_pregao'], format='%Y-%m-%d')
    df_asset['mm5d'] = df_asset.groupby('sigla_acao')['preco_fechamento'].rolling(5).mean().reset_index(0, drop=True)
    df_asset['mm21d'] = df_asset.groupby('sigla_acao')['preco_fechamento'].rolling(21).mean().reset_index(0, drop=True)
    df_asset['preco_fechamento'] = df_asset.groupby('sigla_acao')['preco_fechamento'].shift(-1)
    df_asset.dropna(inplace=True)
    df_asset = df_asset.reset_index(drop=True)

    # Calculate daily returns
    df_asset = calculate_returns(df_asset)
   
    return df_asset


def select_best_features(df_asset):
    features = df_asset.loc[:, ['qtd_negocios', 'preco_maximo', 'preco_minimo', 'mm21d']]
    labels = df_asset['preco_fechamento']
    features_list = ('preco_abertura', 'preco_maximo', 'preco_minimo', 'qtd_negocios', 'volume_negocios', 'mm5d', 'mm21d')
    k_best_features = SelectKBest(k='all')
    k_best_features.fit_transform(features, labels)
    k_best_features_scores = k_best_features.scores_
    raw_pairs = zip(features_list[1:], k_best_features_scores)
    ordered_pairs = list(reversed(sorted(raw_pairs, key=lambda x: x[1])))
    k_best_features_final = dict(ordered_pairs[:15])
    best_features = k_best_features_final.keys()
    print("----------------------------------------------------------------------------")
    print('')
    # print("Melhores features:")
    # print(k_best_features_final)
    return features, labels


def train_neural_network(features, labels):
    scaler = MinMaxScaler()
    X_train_scale = scaler.fit_transform(features)
    rn = MLPRegressor(max_iter=5000)
    rn.fit(X_train_scale, labels)
    valor_novo = features[-10:]
    previsao = scaler.transform(valor_novo)
    pred = rn.predict(previsao)
    
    return pred


def printing_predictions(asset, features, labels, pred):
    df = load_data()
    df_asset = df[df['sigla_acao'] == asset]
    data_pregao_full = df_asset['data_pregao']
    data_pregao = data_pregao_full[-10:]
    res_full = df_asset['preco_fechamento']
    res = res_full[-10:]
    df = pd.DataFrame({'data_pregao': data_pregao, 'valor_real': res, 'previsao': pred})
    df.set_index('data_pregao', inplace=True)
    print('')
    print(asset)
    print(df)


def main():
    df = load_data()

    assets = ['BBDC4', 'BBAS3', 'BPAC11', 'BRSR6', 'SANB11', 'BIDI4', 'ITUB4']
    
    # Lista para armazenar resultados de todas as ações
    results = pd.DataFrame(columns=['Asset', 'Features', 'Prediction'])

    # Dicionario usado para armazenar os daily returns
    returns_dict = {}
    returns_df = pd.DataFrame()

    for asset in assets:
        
        df_asset = filtering_assets(df, asset)

        features, labels = select_best_features(df_asset)

        # atribuindo os daily returns da ação current
        df_asset = calculate_returns(df_asset)
        returns_dict[asset] = df_asset['daily_returns'].values

        # treinando a rede com as features e labels encontradas
        pred = train_neural_network(features, labels)

        # mostrando as previsões
        printing_predictions(asset, features, labels, pred)

        # Criando uma nova linha contendo as ações, features e previsões para cada uma
        row = {'Asset': asset, 'Features': features.columns.tolist(), 'Prediction': pred.tolist()}
        results = results.append(row, ignore_index=True)
 
        returns = df_asset['daily_returns'].tolist()
        returns_df[asset] = df_asset['daily_returns']
        
        # print(f"Daily Returns de {asset}:")
        # print(returns)
        print()
    

    # Calculando covariance matrix
    cov_matrix = risk_models.sample_cov(returns_df)

    # Print da covariance matrix
    print("Covariance Matrix:")
    print(cov_matrix)

    # print(results)


if __name__ == '__main__':
    main()



