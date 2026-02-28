# =============================================
# 1. IMPORTS
# =============================================
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# tentativa segura de importar kagglehub
try:
    import kagglehub
    KAGGLE_AVAILABLE = True
except:
    KAGGLE_AVAILABLE = False

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# =============================================
# CONFIG STREAMLIT
# =============================================
st.set_page_config(layout="wide")
st.title("📊 NASA Turbofan Engine")

# =============================================
# 2. LOAD DATA
# =============================================
@st.cache_data
def load_data():
    if not KAGGLE_AVAILABLE:
        st.error("kagglehub não instalado. Verifique requirements.txt")
        return pd.DataFrame()

    try:
        path = kagglehub.dataset_download(
            "bishals098/nasa-turbofan-engine-degradation-simulation"
        )

        DATA_DIR = Path(path)

        train_path = next(DATA_DIR.rglob("train_FD001.txt"))

        columns = [
            'unit', 'cycle', 'setting1', 'setting2', 'setting3',
            'sensor1', 'sensor2', 'sensor3', 'sensor4', 'sensor5',
            'sensor6', 'sensor7', 'sensor8', 'sensor9', 'sensor10',
            'sensor11', 'sensor12', 'sensor13', 'sensor14', 'sensor15',
            'sensor16', 'sensor17', 'sensor18', 'sensor19', 'sensor20',
            'sensor21'
        ]

        df = pd.read_csv(train_path, sep=r'\s+', header=None, names=columns)

        max_cycles = df.groupby('unit')['cycle'].max()
        df['RUL'] = df.apply(lambda row: max_cycles[row['unit']] - row['cycle'], axis=1)
        df['RUL'] = df['RUL'].clip(upper=125)

        return df

    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()


df_train = load_data()

if df_train.empty:
    st.stop()

# =============================================
# 3. FEATURES
# =============================================
features = [
    'sensor2', 'sensor3', 'sensor4',
    'sensor7', 'sensor11', 'sensor12',
    'sensor14', 'sensor15'
]

# =============================================
# TREINAR MODELO
# =============================================
@st.cache_resource
def train_model(df):
    X = df[features]
    y = df['RUL']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return model, mae, X_test, y_test, y_pred


model, mae, X_test, y_test, y_pred = train_model(df_train)

# =============================================
# MENU
# =============================================
opcao = st.sidebar.radio(
    "📌 Menu:",
    ["Visão Geral", "Sensores", "Correlação", "Manutenção", "Machine Learning"]
)

# =============================================
# FILTRO POR MOTOR
# =============================================
motor = st.sidebar.selectbox(
    "Selecione o motor:",
    ["Todos"] + sorted(df_train['unit'].unique().tolist())
)

df_filtered = df_train if motor == "Todos" else df_train[df_train['unit'] == motor]

# =============================================
# ALERTAS
# =============================================
st.sidebar.markdown("## 🚨 Status dos Motores")

rul_motor = df_train.groupby('unit')['RUL'].min().reset_index()

criticos = rul_motor[rul_motor['RUL'] < 30]
atencao = rul_motor[(rul_motor['RUL'] >= 30) & (rul_motor['RUL'] < 50)]
saudaveis = rul_motor[rul_motor['RUL'] >= 50]

st.sidebar.error(f"🔴 Críticos: {len(criticos)}")
st.sidebar.warning(f"🟡 Atenção: {len(atencao)}")
st.sidebar.success(f"🟢 Saudáveis: {len(saudaveis)}")

# =============================================
# VISÃO GERAL
# =============================================
if opcao == "Visão Geral":
    st.subheader("📊 Informações Gerais")

    col1, col2, col3 = st.columns(3)

    col1.metric("Motores", df_filtered['unit'].nunique())
    col2.metric("Registros", len(df_filtered))
    col3.metric("Sensores", 21)

    st.dataframe(df_filtered.head())
    st.subheader("🚨 Motores Críticos")
    st.dataframe(criticos)

# =============================================
# SENSORES
# =============================================
elif opcao == "Sensores":
    st.subheader("Evolução dos Sensores")

    df_mean = df_filtered.groupby('cycle')[features].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 3))
    for sensor in features:
        ax.plot(df_mean['cycle'], df_mean[sensor], label=sensor)

    ax.legend()
    st.pyplot(fig, use_container_width=True)

# =============================================
# CORRELAÇÃO
# =============================================
elif opcao == "Correlação":
    st.subheader("Correlação com RUL")

    corr = df_filtered[features + ['RUL']].corr()

    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)

    st.pyplot(fig, use_container_width=True)

# =============================================
# MANUTENÇÃO
# =============================================
elif opcao == "Manutenção":
    st.subheader("🔧 Degradação do Motor")

    df_plot = df_filtered.groupby('cycle')['RUL'].mean().reset_index()

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df_plot['cycle'], df_plot['RUL'])

    ax.axhline(50, linestyle='--')
    ax.axhline(30, linestyle='--')

    st.pyplot(fig, use_container_width=True)

# =============================================
# MACHINE LEARNING
# =============================================
elif opcao == "Machine Learning":
    st.subheader("🤖 Previsão de Falha")

    st.metric("Mean Absolute Error", f"{mae:.2f} ciclos")

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.scatter(y_test, y_pred, alpha=0.3)
    ax.set_xlabel("Real")
    ax.set_ylabel("Previsto")

    st.pyplot(fig, use_container_width=True)

    importancias = pd.Series(model.feature_importances_, index=features)
    importancias = importancias.sort_values(ascending=False)

    st.subheader("🔥 Sensores mais relevantes")

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    importancias.plot(kind='bar', ax=ax2)

    st.pyplot(fig2, use_container_width=True)

    df_train['RUL_previsto'] = model.predict(df_train[features])

    risco = df_train.groupby('unit')['RUL_previsto'].min().reset_index()
    criticos_ml = risco[risco['RUL_previsto'] < 30]

    st.subheader("🚨 Motores com risco (ML)")
    st.dataframe(criticos_ml)
