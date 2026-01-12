📑 README — Clientes Perfeitos: Intenção dos Compradores Online
📌 Visão Geral
Este projeto implementa um pipeline de Machine Learning para prever a intenção de compra de clientes online, utilizando o dataset Online Shoppers Purchasing Intention. O objetivo é identificar padrões de comportamento que indicam maior probabilidade de conversão em compra, auxiliando áreas de marketing, vendas e experiência do cliente.
🎯 Objetivos
- Carregar e tratar dados de clientes online.
- Treinar e comparar diferentes modelos de classificação (Regressão Logística, Random Forest e XGBoost).
- Avaliar desempenho com métricas robustas (ROC-AUC).
- Gerar relatórios interpretativos com diagnósticos estatísticos e variáveis mais relevantes.
- Disponibilizar interface interativa via Streamlit para visualização dos resultados.
🛠️ Tecnologias Utilizadas
- Python 3.9+
- Bibliotecas principais:
- pandas, numpy — manipulação de dados
- scikit-learn — pré-processamento e modelos
- xgboost — modelo avançado de boosting
- imbalanced-learn — balanceamento de classes (SMOTE)
- matplotlib — visualização
- statsmodels — diagnósticos estatísticos
- streamlit — interface interativa
📂 Estrutura do Projeto
projeto-semanthix/
│
├── app.py                        # Aplicação principal em Streamlit
├── online_shoppers_intention.csv # Dataset de clientes
├── README.md                     # Documento de referência
└── requirements.txt              # Dependências do projeto


⚙️ Instalação e Execução
- Clonar o repositório
git clone https://github.com/seuusuario/projeto-semanthix.git
cd projeto-semanthix


- Criar ambiente virtual
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows


- Instalar dependências
pip install -r requirements.txt


- Executar aplicação
streamlit run app.py


Acesse no navegador: http://localhost:8501
📊 Funcionalidades
- Carregamento de dados via upload ou arquivo padrão.
- Treinamento automático de três modelos de classificação.
- Comparação de desempenho com métricas ROC-AUC.
- Visualização gráfica das curvas ROC.
- Análise de variáveis relevantes (feature importance).
- Relatório interpretativo com diagnósticos estatísticos e recomendações práticas.
- Download do relatório em formato .txt.
📈 Resultados Esperados
- Melhor modelo identificado: XGBoost (ROC-AUC ≈ 0.926).
- Principais variáveis influenciadoras:
- ValoresPáginas
- Taxas de Saída
- ProductRelated_Duration
- Administrativo
- Relacionado ao Produto
👥 Público-Alvo
- Empresas de e-commerce que desejam aumentar conversão.
- Equipes de marketing e vendas para direcionar campanhas.
- Analistas de dados interessados em modelos preditivos aplicados ao comportamento do consumidor.
📌 Boas Práticas Corporativas
- Código modular e comentado.
- Documentação clara e objetiva.
- Relatórios interpretativos para suporte à decisão.
- Interface amigável para usuários não técnicos.
📜 Licença
Este projeto é distribuído sob a licença MIT. Consulte o arquivo LICENSE para mais detalhes.
