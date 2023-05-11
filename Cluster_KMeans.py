import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.preprocessing import Normalizer
import plotly.express as px
import seaborn as sns
import warnings


warnings.filterwarnings("ignore")
Base_dados = pd.read_csv("https://raw.githubusercontent.com/rafaelduria/Clustering/main/dados.csv", sep=';')

pd.set_option('display.max_columns', None) # coluna
pd.set_option('display.max_rows', None) # linha


#posições estoque
#803 = 80
#801 e 802 =  637
#806 = 467
#810 = 2666
#804  = 2485
#820 = 1945
#807 = 2595
# soma = 10875

"""
Convertendo strig para float
"""

Base_dados['Venda_pallet'] = Base_dados['Venda_pallet'].str.replace(',','.').astype(float)

"""
Removendo colunas 
'Peso'
'Quantidade'
'Data' 
"""

Base_dados.drop(['Peso','Quantidade','Data'], axis=1, inplace=True)

"""
Agrupando Colunas "Venda_pallet" "Frequencia" por Material  
"""

Base_dados['Venda_pallet'] = Base_dados.groupby('Material')["Venda_pallet"].transform(np.sum).round(8)
Base_dados['Frequencia'] = Base_dados.groupby('Material')["Frequencia"].transform(np.sum).round(8)


"""
Retirando valores duplicados
"""

Base_dados = Base_dados.drop_duplicates()

"""
Verificando valores igual zero 
"""

missing = Base_dados.loc[(Base_dados['Venda_pallet'] == 0.0) & (Base_dados['Frequencia'] == 0.0)].count()

"""
Desconsiderando valores igual zero ou menor do que 0
"""

Base_dados = Base_dados.loc[(Base_dados['Venda_pallet'] > 0) & (Base_dados['Frequencia'] > 0) ]


"""
Removendo colunas 'Material'
"""

Base_dados.drop(['Material'], axis=1, inplace=True)




values = Normalizer().fit_transform(Base_dados.values)


kmeans = KMeans(n_clusters=7,n_init='auto',max_iter=300,algorithm='elkan')
y_pred = kmeans.fit_predict(Base_dados.values)
labels = kmeans.labels_


"""
n_clusters=  número de cluster

max_iter= número maximo de interações

algorithm=
  seleciona entre três formatos de cálculo das distâncias:

  • full: padrão dos algoritmos de k-means que calcula a distância de todos os
  pontos com relação ao centro;

  • elkan: converge para os mesmos resultados, mas reduz, a partir de
  triangulações, a quantidade de distâncias calculadas, eliminando muitas etapas
  desnecessárias/redundantes;

  • auto: recorre ao método convencional para conjuntos de dados esparsos, em
  que o algoritmo elkan implementado é incapaz de convergir, e ao método
  elkan para conjuntos de dados densos (SKLEARN, 2020).

"""


"""
Silhouette
MAIOR QUE 0, coeficiente vai de -1 até 1. 
Então, a partir do momento que temos um valor positivo, conseguiremos ter uma ideia se o cluster está bom de acordo com esta métrica ou não.
"""

silhouette = metrics.silhouette_score(values, labels, metric='euclidean')
print('Silhouette_score {:.3f}'.format(silhouette))



"""
Davies Bouldin, 
MAIS PROXIMO DE 0, melhor.
"""

dbs = metrics.davies_bouldin_score(values, labels)
print('Davies_bouldin_score {:.3f}'.format(dbs))

"""
Calinski,
MAIS ALTO melhor.
"""

calinski = metrics.calinski_harabasz_score(values, labels)
print('Calinski_harabasz_score {:.3f}'.format(calinski))



Base_dados['cluster'] = labels
fig = px.scatter(Base_dados, x=Base_dados.Venda_pallet, y=Base_dados.Frequencia, color='cluster')
fig.show()


Base_dados.groupby("cluster").describe().round(2)


#posições estoque
#803 = 80
#801 e 802 =  637
#806 = 467
#810 = 2666
#804  = 2485
#820 = 1945
#807 = 2595
# soma = 10875
