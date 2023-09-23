import pandas as pd
url = 'https://raw.githubusercontent.com/muzanpvp/EXERCICIO09IA/main/diabetes.csv'
base_Treinamento = pd.read_csv(url)

"""Importação da base de teste"""

url_teste = 'https://raw.githubusercontent.com/muzanpvp/EXERCICIO09IA/main/teste.csv'
base_teste = pd.read_csv(url_teste)

"""Importação da base de treinos"""

url_treino = 'https://raw.githubusercontent.com/muzanpvp/EXERCICIO09IA/main/treino.csv'
base_treino = pd.read_csv(url_treino)

"""Salvar os dados de treino e teste em arquivos diferentes"""

df = pd.DataFrame(base_Treinamento.iloc[:615])
df.to_csv(r'treino.csv')

df = pd.DataFrame(base_Treinamento.iloc[615:])
df.to_csv(r'teste.csv')

"""Vizualização das 10 primeiras linhas"""

import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix=base_Treinamento.corr() 
sns.clustermap(corr_matrix,annot=True,fmt=".2f")
plt.title("Correlação entre atributos")
plt.show()

for col in base_Treinamento:
  plt.hist(x = base_Treinamento[col])

sns.countplot(x = base_Treinamento['Outcome']);

plt.hist(x = base_Treinamento['Pregnancies'])

plt.hist(x = base_Treinamento['Glucose'])

plt.hist(x = base_Treinamento['BloodPressure'])

plt.hist(x = base_Treinamento['SkinThickness'])

plt.hist(x = base_Treinamento['Insulin'])

plt.hist(x = base_Treinamento['BMI'])

plt.hist(x = base_Treinamento['DiabetesPedigreeFunction'])

plt.hist(x = base_Treinamento['Age'])

"""Separando atributos de classes"""

attributes = base_Treinamento.iloc[:, :8]
print(attributes)

classes = base_Treinamento.loc[:, ['Outcome']]
print(classes)

"""Verifica se tem valores nulos"""

attributes.isnull().sum()

"""Normaliza os atributos racionais"""

import numpy as np 
from sklearn import preprocessing

# Acurácia: 71.22395833333334
qt = preprocessing.QuantileTransformer() 
normalized_attributes = qt.fit_transform(attributes)

# Acurácia: 73.046875
mas = preprocessing.MaxAbsScaler()
normalized_attributes = mas.fit_transform(attributes)

# Acurácia: 50.390625
nm = preprocessing.Normalizer()
normalized_attributes = nm.fit_transform(attributes)

# Acurácia: 65.88541666666666
ss = preprocessing.StandardScaler()
normalized_attributes = ss.fit_transform(attributes)

# Acurácia: 73.56770833333334
mms = preprocessing.MinMaxScaler()
normalized_attributes = mms.fit_transform(attributes)

"""Separar dados de treino e dados de teste"""

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(normalized_attributes, classes, test_size=0.2, random_state=0, shuffle=False)

x_train.shape, x_test.shape, y_train.shape, y_test.shape

"""Salvar dados de treino e teste nomalizado em arquivo binário"""

import pickle 

with open('treino.pkl', mode='wb') as f:
  pickle.dump([x_train, y_train], f)

with open('teste.pkl', mode='wb') as f:
  pickle.dump([x_test, y_test], f)

"""Treinamento KNN"""

from sklearn.neighbors import KNeighborsClassifier
# Treinamento do Perceptron a partir dos atributos de entrada e classificações
modelo = KNeighborsClassifier(n_neighbors = 3)
modelo.fit(normalized_attributes, classes)

# Acurácia do modelo, que é : 1 - (predições erradas / total de predições)
# Acurácia do modelo: indica uma performance geral do modelo. 
# Dentre todas as classificações, quantas o modelo classificou corretamente;
# (VP+VN)/N
print(f'Acurácia: {modelo.score(normalized_attributes, classes)*100}')

pred = modelo.predict(x_test)

print(f'Esperado: {y_test}, Resultado: {pred}')

"""Mostra a porcentagem de acertos"""

from sklearn.metrics import accuracy_score
accuracy_score(pred, y_test)*100
