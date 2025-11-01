import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(layout="wide", page_title="Aprendizado Supervisionado")


@st.cache_data
def treinar_modelo_regressao(df_dados):
    """Treina um modelo de regressão (Random Forest) para prever a progressão da doença."""
    X = df_dados.drop('progressao_doenca', axis=1)
    y = df_dados['progressao_doenca']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    
    y_pred = modelo.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return modelo, mse, r2, y_test, y_pred

st.title("🎯 Aprendizado Supervisionado: Prevendo a Progressão")

if 'df' not in st.session_state:
    st.error("Os dados não foram carregados. Por favor, volte à Página Inicial primeiro.")
else:
    df = st.session_state['df']
    
    # Treina o modelo
    modelo_reg, mse, r2, y_test, y_pred = treinar_modelo_regressao(df)

    st.markdown("""
    O objetivo aqui é usar os dados dos pacientes (idade, IMC, pressão, etc.) para **prever o valor** da progressão da doença.
    Como nosso alvo (`progressao_doenca`) é um número contínuo, este é um problema de **Regressão**.
    
    Utilizamos um modelo **Random Forest Regressor**.
    """)
    
    st.header("Resultados do Modelo")
    
    col1, col2 = st.columns(2)
    col1.metric("Erro Quadrático Médio (MSE)", f"{mse:.2f}")
    col2.metric("Coeficiente de Determinação (R²)", f"{r2:.2f}")
    
    st.markdown(f"""
    * **MSE (Mean Squared Error):** O erro médio das previsões. Quanto menor, melhor.
    * **R² (R-squared):** Indica o quanto o modelo explica a variabilidade dos dados. Nosso modelo explica **{r2*100:.1f}%** da variância na progressão da doença.
    """)
    
    st.subheader("Previsões vs. Valores Reais (Dados de Teste)")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test, y=y_pred, mode='markers', name='Previsões vs Reais',
                             marker=dict(color='blue', opacity=0.7)))
    fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()], 
                             mode='lines', name='Linha Ideal (y=x)', line=dict(color='red', dash='dash')))
    fig.update_layout(title='Previsões do Modelo vs. Valores Reais',
                      xaxis_title='Valores Reais da Progressão',
                      yaxis_title='Previsões do Modelo')
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("Idealmente, os pontos deveriam estar sobre a linha vermelha. Nosso modelo mostra uma boa tendência.")
    
    st.header("🧪 Teste o Modelo Interativamente")
    st.markdown("Use os sliders abaixo para simular um paciente e ver a previsão do modelo em tempo real.")

    col_slider1, col_slider2 = st.columns(2)
    
    bmi_slider = col_slider1.slider("IMC (bmi)", float(df['bmi'].min()), float(df['bmi'].max()), float(df['bmi'].mean()))
    s5_slider = col_slider1.slider("Soro Sanguíneo s5", float(df['s5'].min()), float(df['s5'].max()), float(df['s5'].mean()))
    bp_slider = col_slider2.slider("Pressão Arterial (bp)", float(df['bp'].min()), float(df['bp'].max()), float(df['bp'].mean()))
    age_slider = col_slider2.slider("Idade (age)", float(df['age'].min()), float(df['age'].max()), float(df['age'].mean()))

    outras_features_media = df.drop(['progressao_doenca', 'bmi', 's5', 'bp', 'age'], axis=1).mean()
    
    paciente_simulado = np.array([
        age_slider,
        df['sex'].mean(), 
        bmi_slider,
        bp_slider,
        outras_features_media['s1'],
        outras_features_media['s2'],
        outras_features_media['s3'],
        outras_features_media['s4'],
        s5_slider,
        outras_features_media['s6']
    ]).reshape(1, -1)
    
    previsao = modelo_reg.predict(paciente_simulado)
    
    st.subheader("Previsão para o Paciente Simulado:")
    st.metric("Valor Previsto de Progressão da Doença", f"{previsao[0]:.2f}")
