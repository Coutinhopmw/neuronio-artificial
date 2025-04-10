from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import re

df = pd.read_csv('biblia_formatada.csv')

def tratar_texto(texto):
    if pd.isnull(texto):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[_]', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto

for coluna in ['livro', 'texto']:
    df[coluna] = df[coluna].astype(str).apply(tratar_texto)

# TF-IDF
corpus = df['texto'].tolist()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# Dados de entrada e saída
X = df_tfidf.iloc[:, :100]  # usar as 100 primeiras colunas (ou todas, se preferir)
df['label'] = df['livro'].apply(lambda x: 0 if x in ['gênesis', 'êxodo', 'levítico'] else 1)
y = df['label']

# Rede Neural
modelo = Sequential()
modelo.add(Dense(units=10, input_shape=(100,), activation='relu'))
modelo.add(Dense(units=1, activation='sigmoid'))

modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
modelo.summary()

# Treinar
modelo.fit(X, y, epochs=10, batch_size=32)
