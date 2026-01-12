
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier

# ------------------ Configuração visual ------------------
st.set_page_config(page_title="Clientes Perfeitos — Online Shoppers", layout="wide")

st.title("🏆 Clientes Perfeitos — Intenção dos Compradores Online")
st.caption("Pipeline otimizado com Revenue como alvo (classificação binária).")

# ------------------ Carregamento de dados ------------------
@st.cache_data
def load_data(path_or_buffer):
    return pd.read_csv(path_or_buffer)

st.subheader("📂 Carregamento de Dados")
uploaded = st.file_uploader("Selecione o arquivo de clientes (CSV)", type=['csv'])
if uploaded is not None:
    df = load_data(uploaded)
    st.success("✅ Arquivo carregado com sucesso")
else:
    df = load_data("online_shoppers_intention.csv")
    st.info("📌 Arquivo padrão carregado do repositório.")

st.write("Formato:", df.shape)
st.divider()

# ------------------ Preparação ------------------
df['Revenue'] = df['Revenue'].astype(int)
X = df.drop(columns=['Revenue'])
y = df['Revenue']

num_cols = X.select_dtypes(include=['float64','int64']).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

numeric_tf = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_tf = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocess = ColumnTransformer([
    ('num', numeric_tf, num_cols),
    ('cat', categorical_tf, cat_cols)
])

# ------------------ Modelos ------------------
smote = SMOTE(random_state=42)
log_reg = LogisticRegression(max_iter=300, solver='liblinear')
rf = RandomForestClassifier(n_estimators=100, max_features='sqrt', random_state=42, n_jobs=-1)
xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

pipe_log = ImbPipeline([('prep', preprocess), ('smote', smote), ('model', log_reg)])
pipe_rf  = ImbPipeline([('prep', preprocess), ('smote', smote), ('model', rf)])
pipe_xgb = ImbPipeline([('prep', preprocess), ('smote', smote), ('model', xgb)])

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
param_grid_rf = {'model__n_estimators': [100], 'model__max_depth': [None], 'model__max_features': ['sqrt']}
grid_rf = GridSearchCV(pipe_rf, param_grid_rf, cv=cv, scoring='roc_auc', n_jobs=-1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
pipe_log.fit(X_train, y_train)
grid_rf.fit(X_train, y_train)
best_rf = grid_rf.best_estimator_
pipe_xgb.fit(X_train, y_train)

# ------------------ Resultados ------------------
y_proba_log = pipe_log.predict_proba(X_test)[:,1]
y_proba_rf  = best_rf.predict_proba(X_test)[:,1]
y_proba_xgb = pipe_xgb.predict_proba(X_test)[:,1]

auc_log = roc_auc_score(y_test, y_proba_log)
auc_rf  = roc_auc_score(y_test, y_proba_rf)
auc_xgb = roc_auc_score(y_test, y_proba_xgb)

scores = {"Regressão Logística": auc_log, "Floresta Aleatória": auc_rf, "XGBoost": auc_xgb}
chosen = max(scores, key=scores.get)

st.subheader("📊 Resultados dos Modelos")
col1, col2, col3, col4 = st.columns(4)
col1.metric("ROC‑AUC — Regressão Logística", f"{auc_log:.3f}")
col2.metric("ROC‑AUC — Floresta Aleatória", f"{auc_rf:.3f}")
col3.metric("ROC‑AUC — XGBoost", f"{auc_xgb:.3f}")
col4.metric("Modelo com melhor desempenho", chosen)

# ------------------ Gráfico das métricas ------------------
st.subheader("📊 Comparação das Métricas ROC-AUC")
fig, ax = plt.subplots(figsize=(8,6))
ax.bar(scores.keys(), scores.values(), color=["#1E88E5", "#43A047", "#F4511E"])
ax.set_ylim(0, 1)
ax.set_ylabel("ROC-AUC")
ax.set_title("Desempenho dos Modelos")
for i, v in enumerate(scores.values()):
    ax.text(i, v + 0.02, f"{v:.3f}", ha='center', fontweight='bold')
st.pyplot(fig)

st.divider()

# ------------------ Curvas ROC ------------------
st.subheader("📈 Curvas ROC")
fig, ax = plt.subplots(figsize=(8,6))
RocCurveDisplay.from_predictions(y_test, y_proba_log, name='Regressão Logística', ax=ax)
RocCurveDisplay.from_predictions(y_test, y_proba_rf, name='Floresta Aleatória', ax=ax)
RocCurveDisplay.from_predictions(y_test, y_proba_xgb, name='XGBoost', ax=ax)
ax.plot([0,1],[0,1],'k--', label='Aleatório')
ax.set_title("Curvas ROC — Comparação dos Modelos")
ax.legend(loc="lower right")
st.pyplot(fig)

st.divider()

# ------------------ Importância das Variáveis ------------------
st.subheader("🌟 Importância das Variáveis (Feature Importance)")

feature_names = []
if len(num_cols) > 0:
    feature_names.extend(num_cols)
if len(cat_cols) > 0:
    encoder = best_rf.named_steps['prep'].named_transformers_['cat'].named_steps['encoder']
    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    feature_names.extend(cat_feature_names)

importances = best_rf.named_steps['model'].feature_importances_

if importances is not None and len(importances) > 0:
    indices = np.argsort(importances)[::-1]
    top_n = min(15, len(importances))
    top_features = [feature_names[i] for i in indices[:top_n]]
    top_importances = importances[indices[:top_n]]

    fig, ax = plt.subplots(figsize=(10,6))
    ax.barh(top_features[::-1], top_importances[::-1], color="#1E88E5")
    ax.set_xlabel("Importância")
    ax.set_title("Top 15 Variáveis mais Relevantes")
    st.pyplot(fig)
else:
    st.write("⚠️ Não foi possível calcular a importância das variáveis.")

st.divider()

# ------------------ Relatório interpretativo ------------------
# ------------------ Relatório interpretativo ------------------
if len(num_cols) > 1:
    X_num = df[num_cols].dropna()
    vif_data = pd.DataFrame()
    vif_data["Variável"] = X_num.columns
    vif_data["VIF"] = [variance_inflation_factor(X_num.values, i) for i in range(len(X_num.columns))]
    vif_mean = vif_data["VIF"].mean()
else:
    vif_mean = None

if importances is not None and len(importances) > 0:
    indices = np.argsort(importances)[::-1]
    top_n = min(5, len(importances))
    top_features = [feature_names[i] for i in indices[:top_n]]
else:
    top_features = []

# Relatório interpretativo
relatorio = f"""
# Relatório de Interpretação — Clientes Perfeitos

## Desempenho dos Modelos
- Regressão Logística: ROC-AUC = {auc_log:.3f}
- Floresta Aleatória: ROC-AUC = {auc_rf:.3f}
- XGBoost: ROC-AUC = {auc_xgb:.3f}
- Melhor modelo escolhido: {chosen}

A análise mostra que o modelo {chosen} apresentou desempenho superior, capturando melhor os padrões de intenção de compra dos clientes online.

## Diagnósticos Estatísticos
- VIF médio: {vif_mean:.2f} (quanto maior, maior a chance de multicolinearidade).

## Variáveis mais Relevantes
Top 5 variáveis mais importantes no modelo Random Forest:
{', '.join(top_features)}

## Conclusão
O pipeline funcionou corretamente:
- Dados carregados e tratados.
- Modelos treinados e comparados.
- Diagnósticos estatísticos aplicados.
- Melhor modelo escolhido com base em ROC-AUC.

O modelo {chosen} é o mais indicado para prever a intenção de compra online neste dataset.

## Recomendações Práticas
- Implementar o modelo {chosen} em produção para prever intenção de compra em tempo real.
- Monitorar o desempenho periodicamente e recalibrar com novos dados.
- Investigar as variáveis mais relevantes para orientar estratégias de marketing e experiência do cliente.
"""

st.subheader("📑 Relatório de Interpretação")
st.markdown(relatorio)
st.download_button("⬇️ Baixar Relatório", relatorio, file_name="relatorio_clientes_perfeitos.txt")