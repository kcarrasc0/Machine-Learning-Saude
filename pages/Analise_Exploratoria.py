import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(layout="wide", page_title="Análise Exploratória")

st.title("📊 Análise Exploratória dos Dados (EDA)")


if 'df' not in st.session_state:
    st.error("Os dados não foram carregados. Por favor, volte à Página Inicial primeiro.")
else:
    df = st.session_state['df']

    st.subheader("Estatísticas Descritivas")
    st.dataframe(df.describe())
    
    st.subheader("Distribuição da Progressão da Doença (Variável Alvo)")
    fig_hist = px.histogram(
        df, x='progressao_doenca', nbins=50, 
        title='Histograma da Progressão da Doença', 
        labels={'progressao_doenca': 'Progressão da Doença'}
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("Vemos que a progressão da doença segue uma distribuição razoavelmente normal, com uma leve inclinação à direita.")
    
    st.subheader("Correlação das Features com a Progressão da Doença")
    corr = df.corr()['progressao_doenca'].drop('progressao_doenca').sort_values(ascending=False)
    
    fig_corr = px.bar(
        corr, x=corr.index, y=corr.values, 
        title='Correlação das Features com a Progressão da Doença',
        labels={'y': 'Coeficiente de Correlação', 'x': 'Feature'}
    )
    st.plotly_chart(fig_corr, use_container_width=True)
    st.markdown("""
    * **`bmi` (IMC)** e **`s5` (possivelmente relacionado a triglicerídeos)** são as features com maior correlação *positiva* com a progressão da doença.
    * **`s3` (possivelmente HDL, o "bom" colesterol)** tem a correlação *negativa* mais forte, indicando que níveis mais altos estão associados a uma menor progressão.
    """)
    
    st.subheader("Relação entre BMI, BP e Progressão da Doença")
    fig_scatter = px.scatter(
        df, x='bmi', y='bp', color='progressao_doenca',
        title='Relação entre IMC, Pressão Arterial e Progressão da Doença',
        labels={'bmi': 'IMC (Padronizado)', 'bp': 'Pressão Arterial (Padronizada)', 'progressao_doenca': 'Progressão'}
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    st.markdown("Pontos mais claros indicam maior progressão da doença. Parece haver uma tendência de que pacientes com IMC e Pressão Arterial mais altos (quadrante superior direito) têm maior progressão.")
