import math
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# CAMADA DE ENTRADA DOS DADOS


df = pd.read_csv('biblia_formatada.csv')
# print(df.head())

# TRATAMENTO DOS DADOS

def tratar_texto(texto):
    if pd.isnull(texto):
        return ""
    
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[_]', '', texto)
    texto = re.sub(r'\s+', ' ', texto)


    return texto


colunas = ['livro','texto']
for coluna in colunas:
    df[coluna] = df[coluna].astype(str).apply(tratar_texto)

df.to_csv('biblia_tratada.csv', index=False)

# VETORIZAÇÃO DO TEXTO

corpus = df['texto'].tolist()

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(corpus)

# print(X.shape)

# print(vectorizer.get_feature_names_out())


df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())


# CAMADA OCULTA

modelo = Sequential()

# Camada densa (camada de neurônios totalmente conectada)
# 10 neurônios, input com 100 features, ativação ReLU


modelo.add(Dense(unit=10, input_shape=(100), activation='relu'))

# CAMADA DE SAÍDA

modelo.add(Dense(units=1, activation='sigmoid'))

# COMPILANDO O MODELO

modelo.compile(optmizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

modelo.summary()





# FUNÇÃO DE ATIVAÇÃO DO NEURÔNIO

# def sigmoid(x):
#     return 1/(1 + math.exp(-x))

# x1 = 0.8
# x2 = 0.9

# w1 = 0.5
# w2 = 0.9

# bias = 0.1

# # Soma ponderada
# z = (x1 * w1) + (x2 * w2) + bias

# # CHAMANDO A FUNÇÃO DE ATIVAÇÃO
# output = sigmoid(z)

# print(f"Saída do neurônio: {output:.4f}")