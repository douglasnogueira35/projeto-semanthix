# =========================================================
# app.py — Semanthix ML Studio (versão otimizada)
# =========================================================

# ---------------- Imports ----------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import io, time

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             RocCurveDisplay)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

import shap
from lime.lime_tabular import LimeTabularExplainer

# ---------------- Configuração da página ----------------
st.set_page_config(page_title="Semanthix ML Studio", layout="wide")
st.title("🚀 Semanthix ML Studio — Otimizado")

# ---------------- Sidebar ----------------
st.sidebar.header("⚙️ Configurações")

target_col = st.sidebar.text_input("Coluna alvo (binária)", value="Compra")
test_size = st.sidebar.slider("Tamanho do teste", 0.1, 0.4, 0.2, 0.05)
random_state = st.sidebar.number_input("Random state", value=42, step=1)
apply_smote = st.sidebar.checkbox("Aplicar SMOTE", value=True)
cv_folds = st.sidebar.slider("Folds CV", 3, 10, 5, 1)

# ---------------- Upload de dados ----------------
@st.cache_data
def load_data(uploaded):
    return pd.read_csv(uploaded)

uploaded = st.file_uploader("📂 Envie o dataset CSV", type="csv")

if uploaded is None:
    st.warning("Envie um arquivo CSV para iniciar.")
    st.stop()

df = load_data(uploaded)
st.success("✅ Dataset carregado com cache")

# ---------------- Tradução opcional ----------------
traducao = {
    "Revenue":"Compra","BounceRates":"TaxaRejeição","ExitRates":"TaxaSaída",
    "PageValues":"ValorPágina","SpecialDay":"DiaEspecial","Month":"Mês",
    "OperatingSystems":"SistemaOperacional","Browser":"Navegador","Region":"Região",
    "TrafficType":"TipoTráfego","VisitorType":"TipoVisitante","Weekend":"FimDeSemana"
}
df = df.rename(columns={k:v for k,v in traducao.items() if k in df.columns})

# ---------------- Pré-processamento com cache ----------------
@st.cache_data
def preprocess(df, target):
    df = df.dropna(subset=[target])
    y = df[target].astype(int)
    X = df.drop(columns=[target])

    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ])

    X_proc = preprocessor.fit_transform(X)
    return X_proc, y, num_cols, cat_cols, preprocessor

X_proc, y, num_cols, cat_cols, preprocessor = preprocess(df, target_col)

# ---------------- Split + SMOTE com cache ----------------
@st.cache_data
def split_balance(X, y, test_size, random_state, apply_smote):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=test_size, random_state=random_state
    )

    if apply_smote:
        sm = SMOTE(random_state=random_state)
        X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, X_test, y_train, y_test

X_train, X_test, y_train, y_test = split_balance(
    X_proc, y, test_size, random_state, apply_smote
)

# ---------------- Treinamento (sob demanda + cache) ----------------
@st.cache_resource
def train_models(X_train, y_train, random_state):
    log_reg = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=random_state)
    xgb = XGBClassifier(
        n_estimators=400, learning_rate=0.05, max_depth=5,
        subsample=0.8, colsample_bytree=0.8,
        random_state=random_state, eval_metric="logloss", n_jobs=-1
    )

    log_reg.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)

    return log_reg, rf, xgb

if st.button("🚀 Treinar modelos"):
    with st.spinner("Treinando modelos..."):
        log_reg, rf, xgb = train_models(X_train, y_train, random_state)
    st.success("Modelos treinados e cacheados")
else:
    st.stop()

# ---------------- Avaliação ----------------
def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    return {
        "Acurácia": accuracy_score(y_test, y_pred),
        "Precisão": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "AUC": roc_auc_score(y_test, y_proba)
    }, y_pred, y_proba

metrics_log, ypl, ppl = evaluate(log_reg, X_test, y_test)
metrics_rf, ypr, ppr = evaluate(rf, X_test, y_test)
metrics_xgb, ypx, ppx = evaluate(xgb, X_test, y_test)

results = pd.DataFrame([
    {"Modelo":"Logistic", **metrics_log},
    {"Modelo":"RandomForest", **metrics_rf},
    {"Modelo":"XGBoost", **metrics_xgb}
])

st.subheader("📊 Comparação de Modelos")
st.dataframe(results)

# ---------------- ROC ----------------
st.subheader("📈 Curvas ROC")
fig, ax = plt.subplots()
RocCurveDisplay.from_predictions(y_test, ppl, name="Logistic", ax=ax)
RocCurveDisplay.from_predictions(y_test, ppr, name="RandomForest", ax=ax)
RocCurveDisplay.from_predictions(y_test, ppx, name="XGBoost", ax=ax)
st.pyplot(fig)

# ---------------- SHAP otimizado ----------------
st.subheader("🧠 SHAP (amostrado)")

@st.cache_resource
def compute_shap(model, X_sample):
    explainer = shap.TreeExplainer(model)
    return explainer(X_sample)

X_shap = X_test[:500]  # limite para performance
explainer = compute_shap(xgb, X_shap)
shap_values = explainer(X_shap)

fig_shap = plt.figure()
shap.summary_plot(shap_values, X_shap, show=False)
st.pyplot(fig_shap)

# ---------------- LIME ----------------
st.subheader("🔍 LIME")

feature_names = np.array(num_cols + list(
    preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)
))

lime_exp = LimeTabularExplainer(
    training_data=X_train,
    feature_names=feature_names,
    class_names=["0","1"],
    mode="classification"
)

idx = st.number_input("Índice da observação", 0, X_test.shape[0]-1, 0)
exp = lime_exp.explain_instance(X_test[idx], rf.predict_proba, num_features=10)
st.write(exp.as_list())

# ---------------- Download ----------------
def to_excel(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
        df.to_excel(writer, index=False)
    return buf.getvalue()

st.download_button("📥 Baixar resultados (Excel)", to_excel(results),
                   file_name="resultados.xlsx",
                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
