# Modelagem Não Supervisionada.

# --- Célula 1: Imports ---
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.datasets import load_diabetes
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- Célula 2: Carregamento dos Dados ---
diabetes = load_diabetes(as_frame=True)
df = pd.concat([diabetes.data, diabetes.target], axis=1)
df.rename(columns={'target': 'progressao_doenca'}, inplace=True)

# Para clusterização, usamos apenas as features (X)
X = df.drop('progressao_doenca', axis=1)

print("Dados carregados.")

# --- Célula 3: Padronização dos Dados ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Dados padronizados.")

# --- Célula 4: Método do Cotovelo (Elbow Method) ---
print("\nCalculando o Método do Cotovelo (Elbow Method)...")
inercias = []
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inercias.append(kmeans.inertia_)

# Plotando
fig_cotovelo = go.Figure()
fig_cotovelo.add_trace(go.Scatter(x=list(k_range), y=inercias, mode='lines+markers'))
fig_cotovelo.update_layout(title='Método do Cotovelo (Elbow Method)',
                  xaxis_title='Número de Clusters (k)',
                  yaxis_title='Inércia')
fig_cotovelo.show()
print("O gráfico sugere k=4 como um bom número de clusters.")

# --- Célula 5: Treinamento do Modelo K-Means ---
k_otimo = 4
print(f"\nTreinando K-Means com k={k_otimo}...")
kmeans = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Adicionar os clusters ao DataFrame original para análise
df_clusterizado = df.copy()
df_clusterizado['cluster'] = clusters.astype(str)

print("Clusters atribuídos.")

# --- Célula 6: Visualização com PCA ---
print("Aplicando PCA para visualização 2D...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df_clusterizado['pca1'] = X_pca[:, 0]
df_clusterizado['pca2'] = X_pca[:, 1]

# Plotando os clusters
fig_pca = px.scatter(
    df_clusterizado, x='pca1', y='pca2', color='cluster',
    title='Clusters de Pacientes (Visualização com PCA)',
    labels={'pca1': 'Componente Principal 1', 'pca2': 'Componente Principal 2'},
    hover_data=['bmi', 'bp', 'age', 'progressao_doenca']
)
fig_pca.show()

# --- Célula 7: Interpretação dos Clusters ---
print("\nAnalisando as médias dos clusters...")
df_clusterizado['cluster'] = pd.to_numeric(df_clusterizado['cluster'])
df_cluster_summary = df_clusterizado.groupby('cluster')[['bmi', 'bp', 's5', 's3', 'progressao_doenca']].mean().reset_index()

print(df_cluster_summary)