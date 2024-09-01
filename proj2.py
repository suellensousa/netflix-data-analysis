import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Dados fornecidos
data = {
    'Nome': ['Ana', 'Bruno', 'Carla', 'Daniel', 'Eduarda', 'Felipe', 'Gabriela', 'Henrique', 'Isabela', 'João'],
    'Idade': [25, 32, 28, 45, 22, 35, 29, 41, 30, 27],
    'Sexo': ['F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M'],
    'Peso (kg)': [55, 85, 62, 90, 48, 78, 70, 95, 60, 80],
    'Altura (m)': [1.65, 1.75, 1.60, 1.80, 1.55, 1.70, 1.68, 1.85, 1.62, 1.77]
}

df = pd.DataFrame(data)

# Calculando o IMC
df['IMC'] = df['Peso (kg)'] / (df['Altura (m)'] ** 2)

# Exibindo o DataFrame
print(df)

# Estatísticas descritivas
print(df.describe())

# Estatísticas descritivas agrupadas por sexo
print(df.groupby('Sexo').describe())

# Análise da distribuição do IMC
print(df['IMC'].describe())

# Visualizações
plt.figure(figsize=(10, 6))
sns.histplot(df['Idade'], kde=True)
plt.title('Distribuição das Idades')
plt.xlabel('Idade')
plt.ylabel('Frequência')
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(x='Altura (m)', y='Peso (kg)', hue='Sexo', data=df, s=100)
plt.title('Peso vs Altura')
plt.xlabel('Altura (m)')
plt.ylabel('Peso (kg)')
plt.legend(title='Sexo')
plt.show()

plt.figure(figsize=(10, 6))
sns.barplot(x='Sexo', y='IMC', data=df)
plt.title('IMC Médio por Sexo')
plt.xlabel('Sexo')
plt.ylabel('IMC')
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(x='Sexo', y='IMC', data=df)
plt.title('Distribuição do IMC por Sexo')
plt.xlabel('Sexo')
plt.ylabel('IMC')
plt.show()