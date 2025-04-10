import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
import numpy as np

# 1. Carregar dados
df = pd.read_csv('biblia_formatada.csv')

# 2. Função para tratamento de texto
def tratar_texto(texto):
    if pd.isnull(texto):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[_]', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto

# 3. Aplicar tratamento
for coluna in ['livro', 'texto']:
    df[coluna] = df[coluna].astype(str).apply(tratar_texto)

# 4. Vetorização TF-IDF
corpus = df['texto'].tolist()
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(corpus)
df_tfidf = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())

# 5. Preparar dados
X = df_tfidf.iloc[:, :1000]  # usar as 100 primeiras colunas
df['label'] = df['livro'].apply(lambda x: 0 if x in ['gênesis', 'êxodo', 'levítico'] else 1)
y = df['label']

# 6. Criar e compilar o modelo
modelo = Sequential()
modelo.add(Dense(units=64, input_shape=(1000,), activation='relu'))
modelo.add(Dense(units=32, activation='relu'))
modelo.add(Dense(units=1, activation='sigmoid'))
modelo.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 7. Treinar modelo
modelo.fit(X, y, epochs=40, batch_size=32)

# 8. Fazer previsões
y_pred_prob = modelo.predict(X)
y_pred_class = (y_pred_prob > 0.5).astype(int)

# 9. Contar e calcular porcentagens

classe_0_pct = (np.sum(y_pred_class == 0) / len(y_pred_class)) * 100
classe_1_pct = (np.sum(y_pred_class == 1) / len(y_pred_class)) * 100

# 10. Gráfico de barras com as porcentagens
plt.figure(figsize=(6, 4))
plt.bar(['Classe 0 (Pentateuco)', 'Classe 1 (Outros)'], [classe_0_pct, classe_1_pct], color=['skyblue', 'salmon'])
plt.ylabel('Porcentagem (%)')
plt.title('Distribuição das Previsões')
plt.ylim(0, 100)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig("grafico_porcentagens.png")
print("Gráfico salvo como 'grafico_porcentagens.png'")