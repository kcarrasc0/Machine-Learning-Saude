import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.set_page_config(layout="wide", page_title="Aprendizado Não Supervisionado")

@st.cache_data
def treinar_modelo_clusterizacao(df_dados):
    """Aplica K-Means para encontrar clusters de pacientes."""

    X = df_dados.drop('progressao_doenca', axis=1)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    inercias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        inercias.append(kmeans.inertia_)
        
    k_otimo = 4
    kmeans = KMeans(n_clusters=k_otimo, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_clusterizado = df_dados.copy()
    df_clusterizado['cluster'] = clusters.astype(str) 
    
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_clusterizado['pca1'] = X_pca[:, 0]
    df_clusterizado['pca2'] = X_pca[:, 1]
    
    return df_clusterizado, inercias

st.title("🧬 Aprendizado Não Supervisionado: Grupos de Pacientes")

if 'df' not in st.session_state:
    st.error("Os dados não foram carregados. Por favor, volte à Página Inicial primeiro.")
else:
    df = st.session_state['df']
    
    # Treina o modelo
    df_cluster, inercias_kmeans = treinar_modelo_clusterizacao(df)
    
    st.markdown("""
    Aqui, o objetivo é diferente. Não queremos prever um valor, mas sim **encontrar grupos (clusters)** de pacientes que sejam semelhantes entre si, com base em suas 10 features. Não usamos a variável `progressao_doenca` 
    para criar os grupos.
    
    Utilizamos o algoritmo **K-Means**.
    """)
    
    st.header("Encontrando o Número Ideal de Clusters (K)")
    st.markdown("Usamos o **Método do Cotovelo (Elbow Method)**. O 'cotovelo' (ponto onde a linha começa a achatar) sugere um bom número de clusters.")
    
    fig_cotovelo = go.Figure()
    fig_cotovelo.add_trace(go.Scatter(x=list(range(1, 11)), y=inercias_kmeans, mode='lines+markers'))
    fig_cotovelo.update_layout(title='Método do Cotovelo (Elbow Method)',
                      xaxis_title='Número de Clusters (k)',
                      yaxis_title='Inércia (Soma das distâncias quadradas)')
    fig_cotovelo.add_vline(x=4, line=dict(color='red', dash='dash'), annotation_text='k=4 (Cotovelo Sugerido)')
    st.plotly_chart(fig_cotovelo, use_container_width=True)
    st.markdown("O gráfico sugere que **k=4** é um bom número de clusters. Vamos usá-lo para agrupar os pacientes.")
    
    st.header("Visualização dos Clusters (com PCA)")
    st.markdown("""
    Como não podemos plotar um gráfico de 10 dimensões, usamos a **Análise de Componentes Principais (PCA)**
    para reduzir as 10 features a apenas 2 componentes (PCA1 e PCA2), preservando o máximo de informação possível.
    """)
    
    fig_pca = px.scatter(
        df_cluster, x='pca1', y='pca2', color='cluster',
        title='Clusters de Pacientes (Visualização com PCA)',
        labels={'pca1': 'Componente Principal 1', 'pca2': 'Componente Principal 2'},
        hover_data=['bmi', 'bp', 'age', 'progressao_doenca']
    )
    st.plotly_chart(fig_pca, use_container_width=True)
    
    st.header("Interpretação dos Clusters")
    st.markdown("O que define cada grupo? Vamos analisar os valores médios das features mais importantes para cada cluster.")
    
    df_cluster['cluster'] = pd.to_numeric(df_cluster['cluster'])
    df_cluster_summary = df_cluster.groupby('cluster')[['bmi', 'bp', 's5', 'progressao_doenca']].mean().reset_index()
    
    st.dataframe(df_cluster_summary.style.background_gradient(cmap='viridis'))
    
    st.markdown("""
    **Insights (Exemplo de Interpretação):**
    * **Cluster 0:** Pacientes com **IMC (`bmi`) e Pressão Arterial (`bp`) baixos**, e também a **menor média de progressão da doença**. Este parece ser o grupo "mais saudável".
    * **Cluster 1:** Pacientes com **IMC (`bmi`) e `s5` muito altos**, e a **pior média de progressão da doença**. Este parece ser o grupo de "alto risco".
    * **Cluster 2:** Um grupo intermediário, com **IMC (`bmi`) e Pressão (`bp`) levemente negativos** (abaixo da média).
    * **Cluster 3:** Pacientes com **Pressão Arterial (`bp`) alta**, mas IMC (`bmi`) próximo da média.
    
    Esta análise de clusterização poderia ajudar os médicos a identificar perfis de pacientes.
    """)
