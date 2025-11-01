import streamlit as st
import pandas as pd
from sklearn.datasets import load_diabetes

st.set_page_config(
    page_title="Análise de Diabetes",
    page_icon="🩺",
    layout="wide"
)

@st.cache_data
def carregar_dados():
    """Carrega o dataset de diabetes do sklearn e o retorna como DataFrame."""
    diabetes = load_diabetes()
    df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
    df['progressao_doenca'] = diabetes.target
    return df

def pagina_inicial():
    st.title("🩺 Análise de Diabetes com Machine Learning")
    st.markdown("""
        Bem-vindo ao projeto final de Machine Learning Aplicado à Saúde.

        Esta aplicação interativa demonstra a aplicação de técnicas de **Aprendizado Supervisionado**
        e **Não Supervisionado** em um conjunto de dados real sobre diabetes.

        O dataset utilizado é o `load_diabetes` da biblioteca Scikit-learn, que contém
        dados de 442 pacientes.

        ### Estrutura da Aplicação
        Use a barra lateral à esquerda para navegar pelas diferentes seções:

        1.  **Análise Exploratória (EDA):** Entendendo as características e correlações dos dados.
        2.  **Aprendizado Supervisionado:** Um modelo de Regressão para prever a progressão da doença.
        3.  **Aprendizado Não Supervisionado:** Um modelo de Clusterização (K-Means) para encontrar perfis de pacientes.

        ---
    """)

    df = carregar_dados()

    st.session_state['df'] = df

    st.subheader("Amostra dos Dados")
    st.dataframe(df.head())

    st.success("Dados carregados com sucesso! Você já pode navegar para as outras páginas.")

if __name__ == "__main__":
    pagina_inicial()

