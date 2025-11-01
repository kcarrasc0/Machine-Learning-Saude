# Análise Exploratória de Dados (EDA).

# --- Célula 1: Imports ---
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.datasets import load_diabetes

# --- Célula 2: Carregamento dos Dados ---
diabetes = load_diabetes(as_frame=True)
df = pd.concat([diabetes.data, diabetes.target], axis=1)
df.rename(columns={'target': 'progressao_doenca'}, inplace=True)

print("Descrição do Dataset:")
print(diabetes.DESCR)
print("\nPrimeiras linhas dos dados:")
print(df.head())

# --- Célula 3: Estatísticas Descritivas ---
print("\nEstatísticas Descritivas:")
print(df.describe())

# --- Célula 4: Histograma da Variável Alvo ---
print("\nAnalisando a distribuição da variável alvo (progressao_doenca)...")
fig_hist = px.histogram(
    df, x='progressao_doenca', nbins=50, 
    title='Histograma da Progressão da Doença', 
    labels={'progressao_doenca': 'Progressão da Doença'}
)
fig_hist.show()

# --- Célula 5: Análise de Correlação ---
print("\nAnalisando a correlação das features com o alvo...")
corr_matrix = df.corr()
corr_target = corr_matrix['progressao_doenca'].drop('progressao_doenca').sort_values(ascending=False)

print(corr_target)

fig_corr = px.bar(
    corr_target, x=corr_target.index, y=corr_target.values, 
    title='Correlação das Features com a Progressão da Doença',
    labels={'y': 'Coeficiente de Correlação', 'x': 'Feature'}
)
fig_corr.show()

# --- Célula 6: Gráfico de Dispersão (Scatter Plot) ---
print("\nAnalisando a relação entre BMI, BP e Progressão...")
fig_scatter = px.scatter(
    df, x='bmi', y='bp', color='progressao_doenca',
    title='Relação entre IMC, Pressão Arterial e Progressão da Doença',
    labels={'bmi': 'IMC (Padronizado)', 'bp': 'Pressão Arterial (Padronizada)', 'progressao_doenca': 'Progressão'}
)
fig_scatter.show()
