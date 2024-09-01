import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('netflix_titles_nov_2019.csv')

print(df.head())

print(df.info())
print(df.isnull().sum())

df = df.dropna(subset=['title', 'listed_in', 'release_year'])

# Extrair o gênero principal da coluna 'listed_in'
df['main_genre'] = df['listed_in'].apply(lambda x: x.split(',')[0])

# Selecionar as colunas para análise
df = df[['release_year', 'main_genre']]

# Codificar os gêneros
label_encoder = LabelEncoder()
df['main_genre_encoded'] = label_encoder.fit_transform(df['main_genre'])

# Separar características e rótulos
X = df[['release_year']]
y = df['main_genre_encoded']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Treinar o modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Visualizar a distribuição dos gêneros
plt.figure(figsize=(10, 6))
sns.countplot(data=df, y='main_genre', order=df['main_genre'].value_counts().index)
plt.title('Distribuição dos Gêneros')
plt.xlabel('Contagem')
plt.ylabel('Gênero')
plt.show()