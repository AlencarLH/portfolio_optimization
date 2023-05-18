import pandas as pd

# Define a posicao de cada campo a ser utilizado em uma lista; e nomeia essas posicoes numa outra lista associada
def read_files(path, name_file, year_date, type_file ):
    
    _file = f'{path}{name_file}.{type_file}'
    
    colspecs = [(2,10),
                (10,12),
                (12,24),
                (27,39),
                (56,69),
                (69,82),
                (82,95),
                (108,121),
                (152,170),
                (170,188)            
    ]
 
    names = ['data_pregao', 'codbdi', 'sigla_acao','nome_acao','preco_abertura','preco_maximo','preco_minimo','preco_fechamento','qtd_negocios','volume_negocios']

    df = pd.read_fwf(_file, colspecs = colspecs, names = names, skiprows =1)
    
    return df

#filtrar ações
def filter_stocks(df):
    df = df [df['codbdi']== 2] #cod lote padrao = 2
    df = df.drop(['codbdi'], 1) #uma vez selecionadas as acoes de lote padrao, a coluna que especifica os outros tipos pode ser dropada
    
    return df
    
#Aterando tipo das data
def parse_date(df):
    df['data_pregao'] = pd.to_datetime(df['data_pregao'], format ='%Y%m%d')
    return df

#divisao dos campos numericos (preços) por 100 para que possuam 2 casas decimais
def parse_values(df):
    df['preco_abertura'] = (df['preco_abertura'] /100).astype(float)
    df['preco_maximo'] = (df['preco_maximo'] /100).astype(float)
    df['preco_minimo'] = (df['preco_minimo'] /100).astype(float)
    df['preco_fechamento'] = (df['preco_fechamento'] /100).astype(float)
    # df['volume_negocios'] = (df['volume_negocios'] /100).astype(int)
    # df['volume_negocios'] = df['volume_negocios'].astype(float).apply(lambda x: '{:.2f}'.format(x))

    return df

#juntando as funções

def concat_files(path, name_file, year_date, type_file, final_file):
    #iterando por todas as datas contando com a possibilidade de existirem varios anos diferentes
    for i , y in enumerate(year_date):
        df = read_files(path, name_file, y, type_file)
        df = filter_stocks(df)
        df = parse_date(df)
        df = parse_values(df)
         
        if i==0:
            df_final = df
        else:
            df_final = pd.concat([df_final, df])
    
    df_final.to_csv(f'{path}//{final_file}', index=False)

    return df_final
            
        
#executando programa de etl

# year_date = ['2018']

# path=f'/home/luis/Documents/b3/'

# name_file='2018'

# type_file ='TXT'

# final_file = 'all_bovespa.csv'

# concat_files( path, name_file, year_date,type_file, final_file)