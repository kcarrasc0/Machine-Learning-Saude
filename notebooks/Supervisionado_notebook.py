# Modelagem Supervisionada.

# --- Célula 1: Imports ---
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Célula 2: Carregamento dos Dados ---
diabetes = load_diabetes(as_frame=True)
df = pd.concat([diabetes.data, diabetes.target], axis=1)
df.rename(columns={'target': 'progressao_doenca'}, inplace=True)

print("Dados carregados.")
print(df.head())

# --- Célula 3: Preparação (Split) dos Dados ---
X = df.drop('progressao_doenca', axis=1)
y = df['progressao_doenca']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Tamanho do treino: {X_train.shape[0]} amostras")
print(f"Tamanho do teste: {X_test.shape[0]} amostras")

# --- Célula 4: Treinamento do Modelo ---
print("\nTreinando o modelo RandomForestRegressor...")
modelo = RandomForestRegressor(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

print("Modelo treinado com sucesso.")

# --- Célula 5: Avaliação do Modelo ---
print("\nAvaliando o modelo nos dados de teste...")
y_pred = modelo.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
print(f"Coeficiente de Determinação (R²): {r2:.2f}")

# --- Célula 6: Visualização dos Resultados ---
print("Plotando Previsões vs. Valores Reais...")
fig = go.Figure()
fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Previsões vs Reais',
                         marker=dict(color='blue', opacity=0.7)))
fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                         mode='lines', name='Linha Ideal (y=x)', line=dict(color='red', dash='dash')))
fig.update_layout(title='Previsões do Modelo vs. Valores Reais',
                  xaxis_title='Valores Reais da Progressão',
                  yaxis_title='Previsões do Modelo')
fig.show()
