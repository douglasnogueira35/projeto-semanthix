# Cliente Perfeito – Dashboard Multi-Modelo de Machine Learning

![Banner](https://img.shields.io/badge/Status-Completo-success?style=flat-square)  

**Autor:** Douglas Silva  
**Data:** 2026-01-12  

---

## 🎯 Visão Geral

O projeto **Cliente Perfeito** é um **dashboard corporativo de Machine Learning**, desenvolvido para analisar padrões de navegação e comportamento de clientes, visando **otimizar estratégias de conversão**.  

O sistema permite:  

- Comparação de múltiplos modelos de ML (RandomForest, XGBoost e Logistic Regression).  
- Seleção automática do melhor modelo baseado em métricas estratégicas.  
- Visualização interativa de métricas, curvas ROC/Precision-Recall e SHAP.  
- Geração de relatórios completos em **TXT** e **PDF**.  
- Interface totalmente em português, moderna e corporativa.  

---

## ⚙️ Funcionalidades Principais

1. **Upload flexível de dados**  
   - Suporte a CSV e Excel (`.xlsx`)  
   - Seleção dinâmica da coluna target  

2. **Preprocessamento automático**  
   - Normalização de variáveis numéricas  
   - One-Hot Encoding para variáveis categóricas  
   - Balanceamento de classes via **SMOTE** (para classificação binária)  

3. **Feature Selection automática**  
   - Seleção das features mais importantes usando RandomForest  
   - Redução do tempo de treinamento e melhoria da interpretabilidade  

4. **Treinamento multi-modelo**  
   - RandomForest, XGBoost e Logistic Regression  
   - Escolha automática do modelo mais adequado para os dados  

5. **Visualização de métricas e gráficos**  
   - KPIs principais: Acurácia, F1-score, ROC AUC, R², RMSE  
   - Curvas interativas ROC e Precision-Recall  
   - Gráficos de importância de features  
   - SHAP para explicabilidade do modelo  

6. **Relatórios executivos**  
   - Texto completo exibido na tela  
   - Download em **TXT** e **PDF**  
   - Insights estratégicos claros e acionáveis  

---

## 🖥️ Tecnologias Utilizadas

- **Python 3.13**  
- **Streamlit** – interface web interativa  
- **Pandas & NumPy** – manipulação de dados  
- **Scikit-learn** – modelagem, métricas e seleção de features  
- **XGBoost** – modelo avançado de classificação e regressão  
- **Imbalanced-learn (SMOTE)** – balanceamento de classes  
- **Plotly** – gráficos interativos e modernos  
- **SHAP** – explicabilidade de modelos de árvore  
- **ReportLab** – geração de PDF profissional  

---

## 🚀 Como Executar

1. Clone o repositório:

```bash
git clone https://github.com/seu-usuario/cliente-perfeito.git
cd cliente-perfeito
