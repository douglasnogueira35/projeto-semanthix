# =====================================================
# IMPORTS
# =====================================================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_squared_error, roc_curve, precision_recall_curve
)
from sklearn.feature_selection import SelectFromModel
from imblearn.over_sampling import SMOTE
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

# =====================================================
# CONFIGURAÇÃO STREAMLIT
# =====================================================
st.set_page_config(
    page_title="Cliente Perfeito | Dashboard Multi-Modelo",
    page_icon="🚀",
    layout="wide"
)

# =====================================================
# SIDEBAR
# =====================================================
st.sidebar.title("🏢 Cliente Perfeito")
st.sidebar.markdown("""
**Dashboard Corporativo de ML – Multi-Modelo**  
- Upload CSV/Excel  
- Comparação de modelos: RandomForest, XGBoost, Logistic Regression  
- Métricas, gráficos e insights estratégicos  
- Download de relatório TXT/PDF
""")
uploaded_file = st.sidebar.file_uploader("📂 Carregar CSV ou Excel", type=["csv","xlsx"])
test_size = st.sidebar.slider("📏 Proporção do conjunto de teste", 0.1, 0.4, 0.2, 0.05)
usar_smote = st.sidebar.checkbox("⚖️ Balancear classes (apenas binário)", True)
mostrar_shap = st.sidebar.checkbox("🧠 Mostrar SHAP (100 amostras)", True)

# =====================================================
# CARREGAMENTO DE DADOS
# =====================================================
@st.cache_data
def carregar_dados(file):
    if hasattr(file, "name"):
        if file.name.lower().endswith(".csv"):
            return pd.read_csv(file)
        elif file.name.lower().endswith(".xlsx"):
            return pd.read_excel(file)
    else:
        if str(file).lower().endswith(".csv"):
            return pd.read_csv(file)
        elif str(file).lower().endswith(".xlsx"):
            return pd.read_excel(file)
    return pd.DataFrame()

df = carregar_dados(uploaded_file) if uploaded_file else carregar_dados("online_shoppers_intention.csv")
if df.empty:
    st.stop()

st.subheader("📊 Visualização Inicial")
st.dataframe(df.head(10))

# =====================================================
# TARGET
# =====================================================
target_col = st.selectbox("Selecione a coluna TARGET", df.columns, index=0)
y_raw = df[target_col]
X = df.drop(columns=[target_col])

if y_raw.nunique() == 2:
    problem_type = "binario"
    y = y_raw.apply(lambda x: 1 if x in [1,"TRUE","True","true"] else 0)
elif y_raw.nunique() <= 10 and y_raw.dtype in [int, object]:
    problem_type = "multiclasse"
    y = y_raw
else:
    problem_type = "regressao"
    y = pd.to_numeric(y_raw, errors="coerce")

# =====================================================
# PREPROCESSAMENTO
# =====================================================
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=42,
    stratify=y if problem_type in ["binario","multiclasse"] else None
)
X_train_p = preprocessor.fit_transform(X_train)
X_test_p = preprocessor.transform(X_test)

if problem_type=="binario" and usar_smote:
    smote = SMOTE(random_state=42)
    X_train_p, y_train = smote.fit_resample(X_train_p, y_train)

# =====================================================
# FEATURE SELECTION AUTOMÁTICA
# =====================================================
if problem_type in ["binario","multiclasse"]:
    fs_model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
else:
    fs_model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
fs_model.fit(X_train_p, y_train)
selector = SelectFromModel(fs_model, threshold="median", prefit=True)
X_train_sel = selector.transform(X_train_p)
X_test_sel = selector.transform(X_test_p)
selected_features = np.array(num_cols + list(preprocessor.named_transformers_["cat"].get_feature_names_out(cat_cols)))[selector.get_support()]

# =====================================================
# TREINAMENTO MULTI-MODELO
# =====================================================
@st.cache_resource
def treinar_modelos(X, y, problem_type):
    modelos = {}
    if problem_type in ["binario","multiclasse"]:
        modelos["RandomForest"] = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1).fit(X, y)
        modelos["XGBoost"] = XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5,
                                           subsample=0.8, colsample_bytree=0.8, random_state=42,
                                           eval_metric="logloss", use_label_encoder=False, n_jobs=-1).fit(X, y)
        modelos["LogisticRegression"] = LogisticRegression(max_iter=1000).fit(X, y)
    else:
        modelos["RandomForest"] = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1).fit(X, y)
        modelos["XGBoost"] = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=5,
                                          subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1).fit(X, y)
    return modelos

modelos = treinar_modelos(X_train_sel, y_train, problem_type)

# =====================================================
# ESCOLHA DO MELHOR MODELO
# =====================================================
resultados = {}
for nome, mod in modelos.items():
    y_pred = mod.predict(X_test_sel)
    if problem_type=="binario":
        y_prob = mod.predict_proba(X_test_sel)[:,1]
        resultados[nome] = roc_auc_score(y_test, y_prob)
    elif problem_type=="multiclasse":
        resultados[nome] = f1_score(y_test, y_pred, average="weighted")
    else:
        resultados[nome] = r2_score(y_test, y_pred)

melhor_modelo_nome = max(resultados, key=resultados.get)
melhor_modelo = modelos[melhor_modelo_nome]

st.subheader(f"🏆 Melhor Modelo: {melhor_modelo_nome} (Score: {resultados[melhor_modelo_nome]:.3f})")

# =====================================================
# MÉTRICAS E KPIs
# =====================================================
def cor_kpi(valor, tipo="percentual"):
    return "normal"

st.subheader("📈 Métricas Principais")
y_pred = melhor_modelo.predict(X_test_sel)
y_prob = melhor_modelo.predict_proba(X_test_sel)[:,1] if problem_type=="binario" else None

if problem_type=="binario":
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc_val = roc_auc_score(y_test, y_prob)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Acurácia", f"{acc:.2%} ✅", delta_color=cor_kpi(acc))
    c2.metric("Precisão", f"{prec:.2%}", delta_color=cor_kpi(prec))
    c3.metric("Recall", f"{rec:.2%}", delta_color=cor_kpi(rec))
    c4.metric("F1-score", f"{f1:.2%}", delta_color=cor_kpi(f1))
    c5.metric("ROC AUC", f"{roc_auc_val:.3f}", delta_color=cor_kpi(roc_auc_val, tipo="valor"))

elif problem_type=="multiclasse":
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    c1,c2 = st.columns(2)
    c1.metric("Acurácia", f"{acc:.2%}", delta_color=cor_kpi(acc))
    c2.metric("F1-score", f"{f1:.2%}", delta_color=cor_kpi(f1))

else:
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    c1,c2 = st.columns(2)
    c1.metric("R²", f"{r2:.3f}", delta_color=cor_kpi(r2, tipo="valor"))
    c2.metric("RMSE", f"{rmse:.3f}", delta_color=cor_kpi(rmse, tipo="valor"))

# =====================================================
# GRÁFICOS
# =====================================================
st.subheader("📊 Gráficos Interativos")
if problem_type=="binario":
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC'))
    roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Random'))
    roc_fig.update_layout(title='Curva ROC', template='plotly_white')
    st.plotly_chart(roc_fig, use_container_width=True)

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_fig = go.Figure()
    pr_fig.add_trace(go.Scatter(x=recall_vals, y=precision_vals, mode='lines', name='Precision-Recall'))
    pr_fig.update_layout(title='Curva Precision-Recall', template='plotly_white')
    st.plotly_chart(pr_fig, use_container_width=True)

imp_df = pd.DataFrame({"Variável": selected_features, "Importância": melhor_modelo.feature_importances_}).sort_values("Importância", ascending=False)
fig_imp = px.bar(imp_df.head(10), x="Importância", y="Variável", orientation="h", template="plotly_white")
fig_imp.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig_imp, use_container_width=True)

# =====================================================
# SHAP
# =====================================================
if mostrar_shap:
    try:
        explainer = shap.TreeExplainer(melhor_modelo)
        shap_values = explainer(X_test_sel[:100])
        shap_imp = np.abs(shap_values.values).mean(axis=0)
        shap_df = pd.DataFrame({"Variável": selected_features, "Impacto Médio": shap_imp}).sort_values("Impacto Médio", ascending=False).head(10)
        fig_shap = px.bar(shap_df, x="Impacto Médio", y="Variável", orientation="h", template="plotly_white")
        fig_shap.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_shap, use_container_width=True)
    except:
        st.warning("SHAP indisponível.")

# =====================================================
# RELATÓRIO FINAL
# =====================================================
relatorio_texto = f"""
RELATÓRIO EXECUTIVO – CLIENTE PERFEITO
Tipo de problema: {problem_type}
Melhor modelo: {melhor_modelo_nome} (Score: {resultados[melhor_modelo_nome]:.3f})
Justificativa: comparação automática de RandomForest, XGBoost e LogisticRegression.

Principais métricas:
"""
if problem_type=="binario":
    relatorio_texto += f"Acurácia: {acc:.2%}\nPrecisão: {prec:.2%}\nRecall: {rec:.2%}\nF1-score: {f1:.2%}\nROC AUC: {roc_auc_val:.3f}\n"
elif problem_type=="multiclasse":
    relatorio_texto += f"Acurácia: {acc:.2%}\nF1-score: {f1:.2%}\n"
else:
    relatorio_texto += f"R²: {r2:.3f}\nRMSE: {rmse:.3f}\n"

relatorio_texto += "\nTop 10 Variáveis / Coeficientes:\n"
for i,row in imp_df.head(10).iterrows():
    relatorio_texto += f"{row['Variável']}: {row['Importância']:.4f}\n"

relatorio_texto += "\nInsights estratégicos:\n- Focar nas features mais importantes.\n- Monitorar padrões de clientes.\n- Ajustar ações de marketing conforme os drivers de conversão.\n"

st.subheader("📄 Relatório Detalhado")
st.text_area("Relatório Completo", relatorio_texto, height=450)
st.download_button("⬇️ Baixar Relatório TXT", relatorio_texto, file_name="relatorio_cliente_perfeito.txt", mime="text/plain")

def gerar_pdf(texto):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer)
    styles = getSampleStyleSheet()
    story = [Paragraph(texto.replace("\n","<br/>"), styles["Normal"])]
    story.append(Spacer(1,12))
    doc.build(story)
    buffer.seek(0)
    return buffer

pdf_buffer = gerar_pdf(relatorio_texto)
st.download_button("⬇️ Baixar Relatório PDF", pdf_buffer, file_name="relatorio_cliente_perfeito.pdf", mime="application/pdf")
