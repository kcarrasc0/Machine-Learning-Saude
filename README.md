![Status](https://img.shields.io/badge/status-em%20desenvolvimento-yellow)

Projeto de Machine Learning aplicado a dados de Saúde, com foco na análise e agrupamento (clustering) de pacientes.

**Objetivo:** Desenvolver um dashboard interativo em Streamlit para aplicar técnicas de aprendizado não supervisionado (K-Means, PCA) e, futuramente, supervisionado, a fim de identificar padrões e insights em um conjunto de dados de pacientes.

**Universitário:**

- Erick Carrasco

**Aplicação**
- [Link para o Dashboard Streamlit (se houver deploy)](https://seu-link-aqui.streamlit.app/)

**Apresentação**
- [Link da Apresentação (se houver)](https://www.canva.com/...)

---

## Estrutura geral de pastas
``
MACHINE-LEARNING-SAUDE/
│
├── 📄 app.py                  # Script principal (página inicial) do Streamlit      
├── 📄 README.md               # Documentação do projeto
├── 📄 requirements.txt         # Lista de bibliotecas Python necessárias
│
├── 📁 .venv/                  # Pasta do ambiente virtual (isolamento)
│   ├── 📁 etc/
│   ├── 📁 Include/
│   ├── 📁 Lib/
│   ├── 📁 Scripts/
│   └── 📁 share/
│
├── 📁 notebooks/               # Seus scripts de análise e modelagem
│   ├── 📄 EDA_notebook.py         # Script para Análise Exploratória dos Dados
│   ├── 📄 Nao_supervisionado_...py  # Script do modelo Não Supervisionado
│   └── 📄 Supervisionado_...py      # Script do modelo Supervisionado
│
└── 📁 pages/                   # Pasta especial do Streamlit para sub-páginas
    ├── 📄 Analise_Explorat...py   # Página do dashboard para Análise Exploratória
    ├── 📄 Aprendizado_Na...py     # Página do dashboard Não Supervisionado (a que mexemos)
    └── 📄 Aprendizado_Su...py     # Página do dashboard Supervisionado
``

---

# Painel de Análise — Não Supervisionado

Esta seção do dashboard, contida no arquivo `pages/Aprendizado_Nao_supervisionado.py`, permite a análise exploratória e o agrupamento de pacientes usando K-Means.

### Resumo técnico dos arquivos

- **`data/dataset_pacientes.csv` (Sugestão):**
  - Dataset (CSV ou TXT) contendo as features dos pacientes (ex: idade, glicose, pressão, etc.) e a feature `progression_disease` mencionada no vídeo.

- **`app.py`:**
  - Página principal (landing page) do dashboard Streamlit.
  - Carrega a introdução ao projeto.
  - O Streamlit gera automaticamente a navegação na barra lateral para as páginas na pasta `pages/`.

- **`pages/Aprendizado_Nao_supervisionado.py`:**
  - Contém toda a lógica para a página de clustering:
  1.  **Carregamento dos Dados:** Lê o dataset de pacientes.
  2.  **Pré-processamento:** (Provavelmente) Aplica `StandardScaler` para normalizar as features antes de alimentar os modelos.
  3.  **Método do Cotovelo (Elbow Method):**
      - Executa o K-Means para um range de `k` (ex: 1 a 10).
      - Plota um gráfico de linha (Inércia vs. Número de Clusters) para ajudar a identificar o 'k' ideal (no nosso caso, k=4).
  4.  **Clusterização K-Means:**
      - Treina o modelo K-Means final com o `k` escolhido (k=4).
      - Adiciona a coluna 'cluster' ao DataFrame.
  5.  **Redução de Dimensionalidade (PCA):**
      - Aplica `PCA(n_components=2)` nos dados normalizados para reduzi-los a 2 dimensões (Componente Principal 1 e 2) para visualização.
  6.  **Visualização dos Clusters:**
      - Plota um gráfico de dispersão (`st.scatter_chart`) dos 2 Componentes Principais, colorindo cada ponto (paciente) pelo seu respectivo cluster.
  7.  **Interpretação dos Clusters:**
      - Calcula a média de cada feature original para cada um dos 4 clusters.
      - Exibe um DataFrame (`st.dataframe`) com esse resumo, usando `.style.background_gradient(cmap='viridis')` para criar um mapa de calor e facilitar a interpretação.

---

## Métricas e Avaliação (Não Supervisionado)

- **Método do Cotovelo:** O gráfico indicou `k=4` como um ponto de inflexão ideal para o número de clusters.
- **Inércia (com k=4):** (Adicionar o valor da inércia aqui, se disponível no seu notebook/script).
- **PCA (Variância Explicada):** (Adicionar a % de variância explicada pelos 2 componentes, se disponível).

---

## Requisitos

- Python 3.9+ (recomendado)
- Um ambiente virtual (`.venv`) é fortemente recomendado.

**Bibliotecas (exemplo de `requirements.txt`):**
---

## Instalação (Windows / PowerShell)

```powershell
# Clone o repositório (se estiver no Git)
git clone [https://seu-repositorio-git-aqui.git](https://seu-repositorio-git-aqui.git)
cd Machine-Learning-Saude

# 1. Criar o ambiente virtual
python -m venv .venv

# 2. Ativar o ambiente virtual
.venv\Scripts\Activate.ps1

# 3. Instalar as dependências
pip install -r requirements.txt

# 4. Execução
streamlit run Home.py
