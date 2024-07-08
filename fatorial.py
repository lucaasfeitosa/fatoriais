#%% pacotes

import pandas as pd
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from factor_analyzer.factor_analyzer import calculate_kmo
import pingouin as pg
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
#%% webscrapping
df_list_ajax = pd.read_html('https://fbref.com/pt/equipes/19c3f8c4/2023-2024/all_comps/Ajax-Estatisticas-Todos-os-campeonatos', attrs={"id":"stats_defense_combined"})
df_list_leon = pd.read_html('https://fbref.com/pt/equipes/fd7dad55/Leon-Estatisticas', attrs={"id":"stats_defense_31"})
df_list_river = pd.read_html('https://fbref.com/pt/equipes/ef99c78c/2024/all_comps/River-Plate-Estatisticas-Todos-os-campeonatos', attrs={"id":"stats_defense_combined"})
df_list_colon = pd.read_html('https://fbref.com/pt/equipes/2d84bb17/2023/all_comps/Colon-Estatisticas-Todos-os-campeonatos', attrs={"id":"stats_defense_combined"})
df_list_san_lorenzo = pd.read_html('https://fbref.com/pt/equipes/66da6009/2024/all_comps/San-Lorenzo-Estatisticas-Todos-os-campeonatos', attrs={"id":"stats_defense_combined"})
df_list_genoa = pd.read_html('https://fbref.com/pt/equipes/658bf2de/2023-2024/all_comps/Genoa-Estatisticas-Todos-os-campeonatos', attrs={"id":"stats_defense_combined"})
df_list_monterrey = pd.read_html('https://fbref.com/pt/equipes/dd5ca9bd/Monterrey-Stats', attrs={"id":"stats_defense_31"})
df_list_mazatlan = pd.read_html('https://fbref.com/pt/equipes/f0297c23/Mazatlan-Stats', attrs={"id":"stats_defense_31"})
df_list_tigres = pd.read_html('https://fbref.com/pt/equipes/d9e1bd51/Tigres-UANL-Stats', attrs={"id":"stats_defense_31"})
df_list_tucuman = pd.read_html('https://fbref.com/pt/equipes/42a1ab8b/2024/all_comps/Atletico-Tucuman-Estatisticas-Todos-os-campeonatos', attrs={"id":"stats_defense_combined"})
df_list_salernitana = pd.read_html('https://fbref.com/pt/equipes/c5577084/2023-2024/all_comps/Salernitana-Estatisticas-Todos-os-campeonatos', attrs={"id":"stats_defense_combined"})
df_list_getafe = pd.read_html('https://fbref.com/pt/equipes/7848bd64/2023-2024/all_comps/Getafe-Estatisticas-Todos-os-campeonatos', attrs={"id":"stats_defense_combined"})
df_list_toluca = pd.read_html('https://fbref.com/pt/equipes/44b88a4e/Toluca-Stats', attrs={"id":"stats_defense_31"})
df_list_pachuca = pd.read_html('https://fbref.com/pt/equipes/1be8d2e3/Pachuca-Stats', attrs={"id":"stats_defense_31"})
df_list_boca = pd.read_html('https://fbref.com/pt/equipes/795ca75e/2024/all_comps/Boca-Juniors-Estatisticas-Todos-os-campeonatos', attrs={"id":"stats_defense_combined"})
df_list_estudiantes = pd.read_html('https://fbref.com/pt/equipes/df734df9/2024/all_comps/Estudiantes-LP-Estatisticas-Todos-os-campeonatos', attrs={"id":"stats_defense_combined"})
df_list_santos = pd.read_html('https://fbref.com/pt/equipes/03b65ba9/Santos-Laguna-Stats', attrs={"id":"stats_defense_31"})
df_list_atlanta = pd.read_html('https://fbref.com/pt/equipes/1ebc1a5b/Atlanta-United-Stats', attrs={"id":"stats_defense_22"})
df_list_sevilla = pd.read_html('https://fbref.com/pt/equipes/ad2be733/2023-2024/all_comps/Sevilla-Estatisticas-Todos-os-campeonatos',attrs={"id":"stats_defense_combined"})
df_list_vasco = pd.read_html('https://fbref.com/pt/equipes/83f55dbe/Vasco-da-Gama-Stats', attrs={"id":"stats_defense_24"})
df_list_banfield = pd.read_html('https://fbref.com/pt/equipes/06c1606c/2024/all_comps/Banfield-Estatisticas-Todos-os-campeonatos', attrs={"id":"stats_defense_combined"})
df_list_puebla = pd.read_html('https://fbref.com/pt/equipes/73fd2313/Puebla-Stats', attrs={"id":"stats_defense_31"})
df_list_sampdoria = pd.read_html('https://fbref.com/pt/equipes/8ff9e3b3/2023-2024/all_comps/Sampdoria-Estatisticas-Todos-os-campeonatos', attrs={"id":"stats_defense_combined"})
df_list_anderlecht = pd.read_html('https://fbref.com/pt/equipes/08ad393c/Anderlecht-Stats', attrs={"id":"stats_defense_37"})
#%% ajustando nome das colunas

# Função para verificar se uma variável é uma lista de dataframes
def is_dataframe_list(var):
    if isinstance(var, list) and all(isinstance(df, pd.DataFrame) for df in var):
        return True
    return False

# obter todas as listas de dataframes do ambiente
df_lists = {name: var for name, var in globals().items() if is_dataframe_list(var)}

# aplicando as transformações a cada dataframe em cada lista
for df_list_name, df_list in df_lists.items():
    # Aplicar a primeira transformação a cada dataframe na lista
    for df in df_list:
        df.columns = ['_'.join(col).strip() for col in df.columns]
 
    # aplicando a segunda transformação a cada dataframe na lista
    for df in df_list:
        df.columns = [col.split('_')[-1].strip() for col in df.columns]


#%% juntando os data frames

all_dfs = []

for df_list_name, df_list in df_lists.items():
    for df in df_list:
        all_dfs.append(df)

df = pd.concat(all_dfs, ignore_index=True)
#%% selecionando zagueiros

zagueiros = df[df['Jogador'].isin(['Léo Pelé', 'Valber Huerta','Adonis Frías','Marcão','Paulo Díaz', 'Facundo Garcés', 'Gastón Silva', 'Alan Matturro', 'Sebastián Vegas', 'Facundo González', 'Samir Santos', 'Nicolás Romero', 'Marco Pellegrino', 'Omar Alderete', 'Federico Pereira', 'Sergio Barreto', 'Jorge Figal','Bruno Amione', 'Dória','Luis Abram','Aaron Quirós'])]
zagueiros = zagueiros.drop(zagueiros.columns[[1,2,3]], axis=1)
zagueiros = zagueiros.drop(zagueiros.columns[-1], axis=1)
zagueiros = zagueiros.drop('Erros', axis=1)
#%% modificando nome das colunas
novos_nomes = {
    2: 'botes_defensivos',
    3: 'disputa_ganha',
    4: 'bote_terço_defensivo',
    5: 'bote_terço_central',
    6: 'bote_terço_ataque',
    7: 'dribles_defendidos',
    8: 'dribles_desafiados',
    9: '%_dribladores_desarmados',
    10: 'desafios_perdidos',
    12: 'chutes_bloqueados',
    13: 'passes_bloqueados',
    14: 'cortes',
    15: 'abordados_mais_interceptações',
    16: 'defesas',
}

for indice, novo_nome in novos_nomes.items():
    zagueiros.columns.values[indice] = novo_nome

zagueiros = zagueiros.drop('dribles_desafiados', axis=1)
zagueiros = zagueiros.drop('bote_terço_ataque', axis=1)
#%% reajustando valores

zagueiros['%_dribladores_desarmados'] = zagueiros['%_dribladores_desarmados']/10
#%% dividindo por 90 minutos

for col in zagueiros.columns:
    if col != '90s' and col != '%_dribladores_desarmados' and col != 'Jogador':
        zagueiros[col] = zagueiros[col] / zagueiros['90s']
        
#%% invertendo base
zagueiros.loc[192, 'desafios_perdidos'] = 0.19
zagueiros['desafios_perdidos'] = 1/zagueiros['desafios_perdidos']
#%% padronizando os dados
# separando variáveis numéricas e não numéricas
numeric_cols = zagueiros.select_dtypes(include=['float64', 'int64']).columns
non_numeric_cols = zagueiros.select_dtypes(exclude=['float64', 'int64']).columns

# padronizando variáveis numéricas
scaler = StandardScaler()
zagueiros_numeric_scaled = scaler.fit_transform(zagueiros[numeric_cols])
zagueiros_numeric_scaled = pd.DataFrame(zagueiros_numeric_scaled, columns=numeric_cols, index=zagueiros.index)

# recombinando as variáveis não numéricas com as variáveis numéricas padronizadas
zagueiros = pd.concat([zagueiros[non_numeric_cols], zagueiros_numeric_scaled], axis=1)

print(zagueiros)
#%% Separando somente as variáveis quantitativas do banco de dados

zagueiros_pca = zagueiros.drop(['Jogador', '90s'], axis=1)
print(zagueiros_pca)
#%% Matriz de correlaçãoes entre as variáveis

matriz_corr = pg.rcorr(zagueiros_pca, method = 'pearson', upper = 'pval', decimals = 4, pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})
print(matriz_corr)

#%% Outra maneira de plotar as mesmas informações

corr = zagueiros_pca.corr()

f, ax = plt.subplots(figsize=(22, 18))

mask = np.triu(np.ones_like(corr, dtype=bool))

cmap = sns.diverging_palette(230, 20, n=256, as_cmap=True)



sns.heatmap(zagueiros_pca.corr(), 
            mask=mask, 
            cmap=cmap, 
            vmax=1, 
            vmin = -.25,
            center=0,
            square=True, 
            linewidths=.5,
            annot = True,
            fmt='.3f', 
            annot_kws={'size': 16},
            cbar_kws={"shrink": .75})

plt.title('Matriz de correlação')
plt.tight_layout()
ax.tick_params(axis = 'x', labelsize = 14)
ax.tick_params(axis = 'y', labelsize = 14)
ax.set_ylim(len(corr))

plt.show()

#%% Teste de Bartlett
bartlett, p_value = calculate_bartlett_sphericity(zagueiros_pca)

print(f'Bartlett statistic: {bartlett}')

print(f'p-value : {p_value}')

#%% Estatística KMO

kmo_all, kmo_model = calculate_kmo(zagueiros_pca)

print(f'kmo_model : {kmo_model}')

#%% Definindo a PCA (procedimento preliminar)

fa = FactorAnalyzer()
fa.fit(zagueiros_pca)


#%% Obtendo os Eigenvalues (autovalores)

ev, v = fa.get_eigenvalues()

print(ev)


#%% Critério de Kaiser

# Verificar autovalores com valores maiores que 1

#%% Parametrizando a PCA para quatro fatores (autovalores > 1)

fa.set_params(n_factors = 4, method = 'principal', rotation = None)
fa.fit(zagueiros_pca)

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
tabela_cargas.index = zagueiros_pca.columns
tabela_cargas

print(tabela_cargas)

#%% Determinando as comunalidades

comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = zagueiros_pca.columns
tabela_comunalidades

print(tabela_comunalidades)

#%% Resultados dos fatores para as observações do dataset (predict)

predict_fatores= pd.DataFrame(fa.transform(zagueiros_pca))
predict_fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(predict_fatores.columns)]

print(predict_fatores)

# Adicionando ao dataset 

zagueiros = pd.concat([zagueiros.reset_index(drop=True), predict_fatores], axis=1)

zagueiros

#%% Identificando os scores fatoriais

scores = fa.weights_

tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = zagueiros_pca.columns
tabela_scores

print(tabela_scores)

#%% Correlação entre os fatores

# A seguir, verifica-se que a correlação entre os fatores é zero (ortogonais)

corr_fator = pg.rcorr(zagueiros[['Fator 1','Fator 2', 'Fator 3', 'Fator 4']], method = 'pearson', upper = 'pval', decimals = 4, pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})
print(corr_fator)

#%% Criando um ranking

zagueiros['Ranking'] = 0

for index, item in enumerate(list(tabela_eigen.index)):
    variancia = tabela_eigen.loc[item]['Variância']

    zagueiros['Ranking'] = zagueiros['Ranking'] + zagueiros[tabela_eigen.index[index]]*variancia
    
print(zagueiros)

#%% renomeando fatores
nomes_fatores = {
    15: 'Habilidade_Defensiva_Geral',
    16: 'Bloqueios_e_Intervenções',
    17: 'Enfretamentos_e_Desafios_Ganhos',
    18: 'Posicionamento_e_Recuperação',
    }

for indice, nomes_fatores in nomes_fatores.items():
    zagueiros.columns.values[indice] = nomes_fatores

zagueiros.columns = zagueiros.columns.str.strip()
zagueiros_fatores = zagueiros[['Jogador', 
                               'Habilidade_Defensiva_Geral',
                               'Bloqueios_e_Intervenções',
                               'Enfretamentos_e_Desafios_Ganhos',
                               'Posicionamento_e_Recuperação', 
                               'Ranking']]
#%% Gráfico das cargas fatoriais e suas variâncias nos componentes principais

import matplotlib.pyplot as plt

plt.figure(figsize=(12,8))

tabela_cargas_chart = tabela_cargas.reset_index()

plt.scatter(tabela_cargas_chart['Fator 1'], tabela_cargas_chart['Fator 2'], s=30)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + 0.05, point['y'], point['val'])

label_point(x = tabela_cargas_chart['Fator 1'],
            y = tabela_cargas_chart['Fator 2'],
            val = tabela_cargas_chart['index'],
            ax = plt.gca()) 

plt.axhline(y=0, color='black', ls='--')
plt.axvline(x=0, color='black', ls='--')
plt.ylim([-1.5,1.5])
plt.xlim([-1.5,1.5])
plt.title(f"{tabela_eigen.shape[0]} componentes principais que explicam {round(tabela_eigen['Variância'].sum()*100,2)}% da variância", fontsize=14)
plt.xlabel(f"PC 1: {round(tabela_eigen.iloc[0]['Variância']*100,2)}% de variância explicada", fontsize=14)
plt.ylabel(f"PC 2: {round(tabela_eigen.iloc[1]['Variância']*100,2)}% de variância explicada", fontsize=14)
plt.show()


#%% Gráfico da variância acumulada dos componentes principais

plt.figure(figsize=(12,8))

plt.title(f"{tabela_eigen.shape[0]} componentes principais que explicam {round(tabela_eigen['Variância'].sum()*100,2)}% da variância", fontsize=14)
sns.barplot(x=tabela_eigen.index, y=tabela_eigen['Variância'], data=tabela_eigen, color='green')
plt.xlabel("Componentes principais", fontsize=14)
plt.ylabel("Porcentagem de variância explicada", fontsize=14)
plt.show()

#%% resetando o banco de dados para segunda análise
#%% selecionando zagueiros

zagueiros = df[df['Jogador'].isin(['Léo Pelé', 'Valber Huerta','Adonis Frías','Marcão','Paulo Díaz', 'Facundo Garcés', 'Gastón Silva', 'Alan Matturro', 'Sebastián Vegas', 'Facundo González', 'Samir Santos', 'Nicolás Romero', 'Marco Pellegrino', 'Omar Alderete', 'Federico Pereira', 'Sergio Barreto', 'Jorge Figal','Bruno Amione', 'Dória','Luis Abram','Aaron Quirós'])]
zagueiros = zagueiros.drop(zagueiros.columns[[1,2,3]], axis=1)
zagueiros = zagueiros.drop(zagueiros.columns[-1], axis=1)
zagueiros = zagueiros.drop('Erros', axis=1)
#%% modificando nome das colunas
novos_nomes = {
    2: 'botes_defensivos',
    3: 'disputa_ganha',
    4: 'bote_terço_defensivo',
    5: 'bote_terço_central',
    6: 'bote_terço_ataque',
    7: 'dribles_defendidos',
    8: 'dribles_desafiados',
    9: '%_dribladores_desarmados',
    10: 'desafios_perdidos',
    12: 'chutes_bloqueados',
    13: 'passes_bloqueados',
    14: 'cortes',
    15: 'abordados_mais_interceptações',
    16: 'defesas',
}

for indice, novo_nome in novos_nomes.items():
    zagueiros.columns.values[indice] = novo_nome

zagueiros = zagueiros.drop('dribles_desafiados', axis=1)
zagueiros = zagueiros.drop('bote_terço_ataque', axis=1)
#%% reajustando valores

zagueiros['%_dribladores_desarmados'] = zagueiros['%_dribladores_desarmados']/10
#%% dividindo por 90 minutos

for col in zagueiros.columns:
    if col != '90s' and col != '%_dribladores_desarmados' and col != 'Jogador':
        zagueiros[col] = zagueiros[col] / zagueiros['90s']
        
#%% invertendo base
zagueiros.loc[192, 'desafios_perdidos'] = 0.33
zagueiros['desafios_perdidos'] = 1/zagueiros['desafios_perdidos']

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
    
    # Excluir colunas com todos os valores iguais a NaN após a normalização
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
#%% Analisando usando o AHP-G

# Analisando todos os jogadores
zagueiros_ahp = zagueiros.drop('90s', axis=1)
resultados, aij_dataframe = calcular_ahp_g(zagueiros_ahp)

# Analisando jogadores comparados
zagueiros_ahp = zagueiros_ahp.drop('AHP-G',axis=1)
resultados_comparado, aij_comparado = calcular_ahp_g(zagueiros_ahp, jogadores=['Bruno Amione', 'Facundo Garcés', 'Léo Pelé'])

#%% criando gráfico de radar

# Extraindo os nomes dos jogadores
jogadores = aij_comparado.index.tolist()

# Extraindo as categorias
categorias = aij_comparado.columns.tolist()

# Número de categorias
num_categorias = len(categorias)

# Obtendo os valores para cada jogador
valores_jogadores = aij_comparado.values

# Número de jogadores
num_jogadores = len(valores_jogadores)

# Calculando os ângulos para cada categoria
angulos = np.linspace(0, 2 * np.pi, num_categorias, endpoint=False).tolist()

# Fechar o gráfico
valores_jogadores = np.concatenate((valores_jogadores, valores_jogadores[:,[0]]), axis=1)
angulos += angulos[:1]

# Plotar o gráfico de radar para cada jogador
fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

for i in range(num_jogadores):
    ax.plot(angulos, valores_jogadores[i], linewidth=2, label=jogadores[i])

# Definindo o título
ax.set_title('Comparação entre Jogadores', size=20, color='black', y=1.1)

# Definindo as etiquetas das categorias
ax.set_xticks(angulos[:-1])
ax.set_xticklabels(categorias, fontsize=12)

# Adicionando uma legenda
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

# Mostrar o gráfico
plt.show()

#%% gráfico de dispersão

# Definindo um tamanho mínimo para os pontos para garantir que sejam visíveis
min_size = 50
point_size = zagueiros_fatores['Enfretamentos_e_Desafios_Ganhos'] * 500
point_size = np.maximum(point_size, min_size)

# Criando a figura e os eixos
fig, ax = plt.subplots(figsize=(10, 6))

# Criando o gráfico de dispersão
scatter = ax.scatter(
    x=zagueiros_fatores['Habilidade_Defensiva_Geral'],
    y=zagueiros_fatores['Bloqueios_e_Intervenções'],
    s=point_size,
    c=zagueiros_fatores['Posicionamento_e_Recuperação'],
    cmap='viridis',  # Paleta de cores
    alpha=0.6,
    edgecolors="w",
    linewidth=0.5
)

# Adicionando os nomes dos jogadores aos pontos
for i, row in zagueiros_fatores.iterrows():
    ax.text(row['Habilidade_Defensiva_Geral'], row['Bloqueios_e_Intervenções'], row['Jogador'], fontsize=9, ha='right')

# Adicionando uma barra de cor
cbar = plt.colorbar(scatter)
cbar.set_label('Fator 4 - Posicionamento e Recuperação')

## Adicionando uma anotação sobre o tamanho dos pontos
ax.annotate('O tamanho dos pontos indica o valor do Fator 3', 
            xy=(0.95, 0.95), xycoords='axes fraction', fontsize=10, ha='right')

# Configurações dos eixos
ax.set_xlabel('Fator 1 - Habilidade Defensiva Geral')
ax.set_ylabel('Fator 2 - Bloqueios e Intervenções')
ax.set_title('Gráfico de Dispersão dos Fatores')

# Exibe o gráfico
plt.show()

#%% gráfico rankings
# Definindo a coluna 'Jogador' como índice
zagueiros_fatores.set_index('Jogador', inplace=True)
resultados.set_index('Jogador', inplace=True)

# Verificando a ordem dos índices para garantir que estejam alinhados
zagueiros_fatores = zagueiros_fatores.loc[resultados.index]

# Criando a figura e os eixos
fig, ax = plt.subplots(figsize=(10, 6))

# Criando o gráfico de dispersão
scatter = ax.scatter(
    x=zagueiros_fatores['Ranking'],
    y=resultados['AHP-G'],
    edgecolors="w",
    linewidth=0.5
)

# Adicionando os nomes dos jogadores aos pontos
for jogador, row in zagueiros_fatores.iterrows():
    ax.text(row['Ranking'], resultados.loc[jogador]['AHP-G'], jogador, fontsize=9, ha='right')
    
# Configurações dos eixos
ax.set_xlabel('Ranking Análise Fatorial')
ax.set_ylabel('Ranking AHP-G')
ax.set_title('Gráfico de Dispersão dos Rankings')

# Exibe o gráfico
plt.show()

#%% Matriz de dispersão para cada um dos pares de atributos dos dados

scatter_matrix = pd.plotting.scatter_matrix(zagueiros_fatores[['Habilidade_Defensiva_Geral',
                                              'Bloqueios_e_Intervenções',
                                              'Enfretamentos_e_Desafios_Ganhos',
                                              'Posicionamento_e_Recuperação']], alpha = 0.9, figsize = (16,10), diagonal = 'kde')
# Exibe o gráfico
plt.show()
