# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 11:11:57 2024

@author: LUCAS MARQUES
"""


#%% Carregando os pacotes necessários
import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import pingouin as pg
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
#%% Importando o banco de dados e invertando as bases das variáveis negativas

segundo_volante = pd.read_excel('C:\\Users\\LUCAS MARQUES\\Downloads\\segundo_volante.xlsx')

# Lista de colunas a serem invertidas
colunas_para_inverter = ['Dribles Sofridos', 'Passes Errados', 'Perda de Posse de Bola', 'Faltas', 'Desarmes Sofridos']

# Aplicando uma função para inverter as bases dessas variáveis
for coluna in colunas_para_inverter:
    segundo_volante[coluna] = 1 / segundo_volante[coluna]
print(segundo_volante)

#%% Informações sobre as variáveis

print(segundo_volante.info())

print(segundo_volante.describe())

#%% padronizando os dados
# separando variáveis numéricas e não numéricas
numeric_cols = segundo_volante.select_dtypes(include=['float64', 'int64']).columns
non_numeric_cols = segundo_volante.select_dtypes(exclude=['float64', 'int64']).columns

# padronizando variáveis numéricas
scaler = StandardScaler()
sv_numeric_scaled = scaler.fit_transform(segundo_volante[numeric_cols])
sv_numeric_scaled = pd.DataFrame(sv_numeric_scaled, columns=numeric_cols, index=segundo_volante.index)

# recombinando as variáveis não numéricas com as variáveis numéricas padronizadas
segundo_volante = pd.concat([segundo_volante[non_numeric_cols], sv_numeric_scaled], axis=1)

#%% Matriz de correlaçãoes entre as variáveis
sv_pca = segundo_volante.drop('Jogador', axis=1)
matriz_corr = pg.rcorr(sv_pca, method = 'pearson', upper = 'pval', decimals = 4, pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})
print(matriz_corr)

#%% Teste de Bartlett

bartlett, p_value = calculate_bartlett_sphericity(sv_pca)

print(f'Bartlett statistic: {bartlett}')

print(f'p-value : {p_value}')


#%% Estatística KMO

kmo_all, kmo_model = calculate_kmo(sv_pca)

print(f'kmo_model : {kmo_model}')

#%% retirando algumas variaveis com baixa correlação pois os dados não passaram nos testes
colunas_para_remover = [
    'Passes para o Assistente', 'Desarmes', 'Dribles Sofridos', 'Duelos Aéreos Ganhos %', 
    'Botes Terço Central', 'Botes Terço Ataque', 'Faltas', '% de Conclusão de Passes Longos', 
    'Acerto no Cruzamento %', 'Carregadas Progressivas'
]

# Removendo as colunas em uma única operação
sv_pca = sv_pca.drop(columns=colunas_para_remover)

#%% Teste de Bartlett

bartlett, p_value = calculate_bartlett_sphericity(sv_pca)

print(f'Bartlett statistic: {bartlett}')

print(f'p-value : {p_value}')


#%% Estatística KMO

kmo_all, kmo_model = calculate_kmo(sv_pca)

print(f'kmo_model : {kmo_model}')


#%% Definindo a PCA (procedimento preliminar)

fa = FactorAnalyzer()
fa.fit(sv_pca)


#%% Obtendo os Eigenvalues (autovalores)

ev, v = fa.get_eigenvalues()

print(ev)

#%% Critério de Kaiser

# Verificar autovalores com valores maiores que 1
# Existem 7 componentes acima de 1

#%% Parametrizando a PCA para 7 fatores (autovalores > 1)

fa.set_params(n_factors = 7, method = 'principal', rotation = None)
fa.fit(sv_pca)


#%% Eigenvalues, variâncias e variâncias acumulada

eigen_fatores = fa.get_factor_variance()
eigen_fatores

tabela_eigen = pd.DataFrame(eigen_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

#%% Determinando as cargas fatoriais

cargas_fatores = fa.loadings_

tabela_cargas = pd.DataFrame(cargas_fatores)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = sv_pca.columns
tabela_cargas

print(tabela_cargas)

#%% Determinando as comunalidades

comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = sv_pca.columns
tabela_comunalidades

print(tabela_comunalidades)

#%% Resultados dos fatores para as observações do dataset (predict)

predict_fatores= pd.DataFrame(fa.transform(sv_pca))
predict_fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(predict_fatores.columns)]

print(predict_fatores)

# Adicionando ao dataset 

segundo_volante = pd.concat([segundo_volante.reset_index(drop=True), predict_fatores], axis=1)

segundo_volante

#%% Identificando os scores fatoriais

scores = fa.weights_

tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = sv_pca.columns
tabela_scores

print(tabela_scores)

#%% Correlação entre os fatores

# A seguir, verifica-se que a correlação entre os fatores é zero (ortogonais)

corr_fator = pg.rcorr(segundo_volante[['Fator 1','Fator 2', 'Fator 3', 'Fator 4',
                              'Fator 5', 'Fator 6', 'Fator 7']], method = 'pearson', upper = 'pval', decimals = 4, pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})
print(corr_fator)

#%% Criando um ranking

segundo_volante['Ranking'] = 0

for index, item in enumerate(list(tabela_eigen.index)):
    variancia = tabela_eigen.loc[item]['Variância']

    segundo_volante['Ranking'] = segundo_volante['Ranking'] + segundo_volante[tabela_eigen.index[index]]*variancia
    
print(segundo_volante)

#%% AHP-G

# carregando uma nova base para realizar outra análise a partir do método AHP-G
segundo_volante2 = pd.read_excel('C:\\Users\\LUCAS MARQUES\\Downloads\\segundo_volante.xlsx')
# Lista de colunas a serem invertidas
colunas_para_inverter = ['Dribles Sofridos', 'Passes Errados', 'Perda de Posse de Bola', 'Faltas', 'Desarmes Sofridos']

# Aplicando uma função para inverter as bases dessas variáveis
for coluna in colunas_para_inverter:
    segundo_volante2[coluna] = 1 / segundo_volante2[coluna]
print(segundo_volante2)


#%% função para calcular aij

def calcular_ahp_g(df, jogadores=None, **kwargs):
    colunas = kwargs.get('colunas', None)
      
         
    # Se nenhum argumento for passado para colunas, selecionar todas as colunas do DataFrame
    if colunas is None:
        colunas = df.select_dtypes(include=[np.number]).columns
    
    # Filtrando o DataFrame pelos jogadores especificados ou selecionar todos se nenhum for especificado
    if jogadores is not None:
        if isinstance(jogadores, str):
            jogadores = [jogadores]  # Converte para lista se for apenas um jogador
        if len(jogadores) == 1:  # Caso apenas um jogador seja especificado
            df_temp_jogador = df[df['Jogador'] == jogadores[0]]
            df_temp_outros = df[df['Jogador'] != jogadores[0]]
            df = pd.concat([df_temp_jogador, df_temp_outros])
        else:  # Caso mais de um jogador seja especificado
            df = df[df['Jogador'].isin(jogadores)]
    
    # Passo 1: Selecionando apenas as colunas numéricas relevantes para os cálculos
    colunas_numericas = df.select_dtypes(include=[np.number]).columns.intersection(colunas)
    df_relevante = df[colunas_numericas]
    
    # Passo 2: Normalização dos dados
    df_normalizado = df_relevante.div(df_relevante.sum())
    
    # Excluir colunas com todos os valores iguais a NaN após a normalização, se existirem
    df_normalizado = df_normalizado.dropna(axis=1, how='all')
    
    # Adiciona a coluna 'Jogador' ao DataFrame normalizado
    df_normalizado['Jogador'] = df['Jogador']

    # Passo 3: Cálculo do desvio padrão dividido pela média
    std_mean_ratio = df_normalizado.drop(columns='Jogador').std() / df_normalizado.drop(columns='Jogador').mean()

    # Passo 4: Cálculo do fator gaussiano normalizado
    gaussian_factors = std_mean_ratio / std_mean_ratio.sum()

    # Passo 5: Cálculo do AHP-G para cada jogador
    ahp_g = np.dot(df_normalizado.drop(columns='Jogador').values, gaussian_factors.values)

    # Adicionando os valores AHP-G
    df.loc[:, 'AHP-G'] = ahp_g
    
    # Cria um DataFrame apenas com as colunas 'Jogador' e 'AHP-G'
    resultados = df[['Jogador', 'AHP-G']]
    
    # Ordena os resultados pelo AHP-G em ordem decrescente
    resultados = resultados.sort_values(by='AHP-G', ascending=False)
    
    # Adiciona uma coluna de ranking
    resultados['Rank'] = resultados.reset_index().index + 1
    
    # Define a coluna 'Jogador' como índice do DataFrame para os aij
    aij_dataframe = df_normalizado.set_index('Jogador')
    
    # Seleciona apenas o jogador especificado
    if jogadores is not None:
        resultados = resultados[resultados['Jogador'].isin(jogadores)]
        aij_dataframe = aij_dataframe.loc[jogadores]

    return resultados, aij_dataframe

#%% calculando o resultado da AHP-G
resultados, aij_dataframe = calcular_ahp_g(segundo_volante2)

#%% calculando a média dos rankings

segundo_volante['Média Rankings'] = (segundo_volante['Ranking'] + segundo_volante2['AHP-G'])/2

#%% gráfico de dispersão

# Definindo um tamanho mínimo para os pontos para garantir que sejam visíveis
min_size = 50
point_size = segundo_volante['Fator 3'] * 500
point_size = np.maximum(point_size, min_size)

# Criando a figura e os eixos
fig, ax = plt.subplots(figsize=(10, 6))

# Criando o gráfico de dispersão
scatter = ax.scatter(
    x=segundo_volante['Fator 1'],
    y=segundo_volante['Fator 2'],
    s=point_size,
    c=segundo_volante['Fator 4'],
    cmap='viridis',  # Paleta de cores
    alpha=0.6,
    edgecolors="w",
    linewidth=0.5
)

# Adicionando os nomes dos jogadores aos pontos
for i, row in segundo_volante.iterrows():
    ax.text(row['Fator 1'], row['Fator 2'], row['Jogador'], fontsize=9, ha='right')

# Adicionando uma barra de cor
cbar = plt.colorbar(scatter)
cbar.set_label('Fator 4')

## Adicionando uma anotação sobre o tamanho dos pontos
ax.annotate('O tamanho dos pontos indica o valor do Fator 3', 
            xy=(0.95, 0.95), xycoords='axes fraction', fontsize=10, ha='right')

# Configurações dos eixos
ax.set_xlabel('Fator 1')
ax.set_ylabel('Fator 2')
ax.set_title('Gráfico de Dispersão dos Fatores')

# Exibe o gráfico
plt.show()

#%% gráfico rankings
# Definindo a coluna 'Jogador' como índice
segundo_volante.set_index('Jogador', inplace=True)
resultados.set_index('Jogador', inplace=True)

# Verificando a ordem dos índices para garantir que estejam alinhados
segundo_volante = segundo_volante.loc[resultados.index]

# Criando a figura e os eixos
fig, ax = plt.subplots(figsize=(10, 6))

# Criando o gráfico de dispersão
scatter = ax.scatter(
    x=segundo_volante['Ranking'],
    y=resultados['AHP-G'],
    edgecolors="w",
    linewidth=0.5
)

# Adicionando os nomes dos jogadores aos pontos
for jogador, row in segundo_volante.iterrows():
    ax.text(row['Ranking'], resultados.loc[jogador]['AHP-G'], jogador, fontsize=9, ha='right')
    
# Configurações dos eixos
ax.set_xlabel('Ranking Análise Fatorial')
ax.set_ylabel('Ranking AHP-G')
ax.set_title('Gráfico de Dispersão dos Rankings')

# Exibe o gráfico
plt.show()


#%% Carregando objeto que será utilizado pra análise de clusterização
# são reinseridas as variáveis retiradas anteriormente
segundo_volante.reset_index('Jogador', inplace=True)
sv_cluster = segundo_volante[['Jogador', 'Passes para o Assistente', 'Desarmes', 
                              'Dribles Sofridos', 'Duelos Aéreos Ganhos %','Botes Terço Central',
                              'Botes Terço Ataque', 'Faltas', '% de Conclusão de Passes Longos',
                              'Acerto no Cruzamento %', 'Carregadas Progressivas', 'Fator 1', 
                              'Fator 2','Fator 3', 'Fator 4', 
                              'Fator 5', 'Fator 6', 'Fator 7']]
#%% criando um gráfico com o método da silhueta para determinar o número ótimo de clusters
from sklearn.metrics import silhouette_score
silhueta = []
I = range(2,30) 
for i in I: 
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100).fit(sv_cluster.drop('Jogador', axis=1))
    silhueta.append(silhouette_score(sv_cluster.drop('Jogador', axis=1), kmeansSil.labels_))

plt.figure(figsize=(16,8))
plt.plot(range(2, 30), silhueta, color = 'purple', marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
plt.title('Método da Silhueta', fontsize=16)
plt.axvline(x = silhueta.index(max(silhueta))+2, linestyle = 'dotted', color = 'red') 
plt.show()

#%% Cluster Não Hierárquico K-means

# O método da silhueta definiu como 2 clusters o número ótimo
# No entanto ficaram muitos jogadores no cluster do Paulinho
# Para reduzir esse número vamos considerar 3 clusters

kmeans_final = KMeans(n_clusters = 3, init = 'random', random_state=100).fit(sv_cluster.drop('Jogador',axis=1))

# Gerando a variável para identificarmos os clusters gerados

kmeans_clusters = kmeans_final.labels_
sv_cluster['cluster_kmeans'] = kmeans_clusters
sv_cluster['Ranking'] = segundo_volante['Ranking']

#%% Criando o dataframe apenas com os jogadores do cluster Paulinho

cluster_paulinho = sv_cluster[sv_cluster['cluster_kmeans'] == 0]

# Gerando uma lista dos jogadores do cluster_paulinho
jogadores_cluster_paulinho = cluster_paulinho['Jogador'].tolist()

# Filtrando no objeto segundo_volante2 para obter apenas os jogadores do cluster_paulinho
# Isso está sendo feito para realizar uma nova AHP-G, dessa vez apenas nos jogadores desse cluster
# Um número menor de jogadores gera um novo resultado da AHP-G
paulinho_ahp = segundo_volante2[segundo_volante2['Jogador'].isin(jogadores_cluster_paulinho)]

resultados_paulinho, aij_dataframe_paulinho = calcular_ahp_g(paulinho_ahp.drop('AHP-G', axis=1))

#%% Juntando tudo em um só dataframe e calculando a média dos rankings


cluster_paulinho = cluster_paulinho.merge(resultados_paulinho[['Jogador', 'AHP-G', 'Rank']], on='Jogador', how='left')
cluster_paulinho['Média Rankings'] = (cluster_paulinho['Ranking'] + cluster_paulinho['AHP-G'])/2
cluster_paulinho = cluster_paulinho.set_index('Jogador')

#%% gráfico de dispersão

# Definindo um tamanho mínimo para os pontos para garantir que sejam visíveis
min_size = 50
point_size = cluster_paulinho['Fator 3'] * 500
point_size = np.maximum(point_size, min_size)

# Criando a figura e os eixos
fig, ax = plt.subplots(figsize=(10, 6))

# Criando o gráfico de dispersão
scatter = ax.scatter(
    x=cluster_paulinho['Fator 1'],
    y=cluster_paulinho['Fator 2'],
    s=point_size,
    c=cluster_paulinho['Fator 4'],
    cmap='viridis',  # Paleta de cores
    alpha=0.6,
    edgecolors="w",
    linewidth=0.5
)

# Adicionando os nomes dos jogadores aos pontos
for jogador, row in cluster_paulinho.iterrows():
    ax.text(row['Fator 1'], row['Fator 2'], jogador, fontsize=9, ha='right')

# Adicionando uma barra de cor
cbar = plt.colorbar(scatter)
cbar.set_label('Fator 4')

## Adicionando uma anotação sobre o tamanho dos pontos
ax.annotate('O tamanho dos pontos indica o valor do Fator 3', 
            xy=(0.95, 0.75), xycoords='axes fraction', fontsize=10, ha='right')

# Configurações dos eixos
ax.set_xlabel('Fator 1')
ax.set_ylabel('Fator 2')
ax.set_title('Gráfico de Dispersão dos Fatores')

# Exibe o gráfico
plt.show()

#%% gráfico rankings

# Criando a figura e os eixos
fig, ax = plt.subplots(figsize=(10, 6))

# Criando o gráfico de dispersão
scatter = ax.scatter(
    x=cluster_paulinho['Ranking'],
    y=cluster_paulinho['AHP-G'],
    edgecolors="w",
    linewidth=0.5
)

# Adicionando os nomes dos jogadores aos pontos
for jogador, row in cluster_paulinho.iterrows():
    ax.text(row['Ranking'],row['AHP-G'], jogador, fontsize=9, ha='right')
    
# Configurações dos eixos
ax.set_xlabel('Ranking Análise Fatorial')
ax.set_ylabel('Ranking AHP-G')
ax.set_title('Gráfico de Dispersão dos Rankings')

# Exibe o gráfico
plt.show()

#%% Matriz de dispersão para cada um dos pares de atributos dos dados



# Definindo uma paleta de cores
palette = sns.color_palette("husl", len(cluster_paulinho.index))


# Criando o DataFrame para a matriz de dispersão
cluster_paulinho_plot = cluster_paulinho[['Fator 1', 'Fator 2', 'Fator 3', 'Fator 4', 'Fator 5', 'Fator 6', 'Fator 7', 'Jogador']]

# Criando a matriz de dispersão com seaborn
sns.pairplot(cluster_paulinho_plot, hue='Jogador', palette=palette, plot_kws={'alpha':0.9})

# Exibindo o gráfico
plt.show()

#%% criando um objeto definindo Jogador como índice apenas para visualizar melhor o data frame
# criando também data frames separados para cada fator para verificar individualmente as cargas fatoriais de cada um
segundo_volante3 = segundo_volante.set_index('Jogador')
fator1 = tabela_cargas['Fator 1']
fator2 = tabela_cargas['Fator 2']
fator3 = tabela_cargas['Fator 3']
fator4 = tabela_cargas['Fator 4']
fator5 = tabela_cargas['Fator 5']
fator6 = tabela_cargas['Fator 6']
fator7 = tabela_cargas['Fator 7']