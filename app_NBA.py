import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ============================
# CONFIGURACIÓN GENERAL
# ============================

st.set_page_config(
    page_title="NBA Analytics Dashboard",
    layout="wide",
    page_icon="🏀"
)

# PALETA DE COLORES PROPIA
COLOR_BG = "#012E40"
COLOR_ACCENT = "#F28705"
COLOR_1 = "#025159"
COLOR_2 = "#038C8C"
COLOR_3 = "#03A696"

st.markdown(
    f"""
    <style>
    .title {{
        font-size: 50px;
        text-align: center;
        font-weight: bold;
        color: {COLOR_ACCENT};
    }}
    body {{
        background-color: {COLOR_BG};
    }}
    .stApp {{
        background-color: {COLOR_BG};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("<h1 class='title'>🏀 NBA Performance Analytics</h1>", unsafe_allow_html=True)

st.markdown(
    f"""
    <h4 style="color:{COLOR_3}; text-align:center;">
        Análisis avanzado de rendimiento, clusters, RMO y predicción de desempeño
    </h4>
    """,
    unsafe_allow_html=True,
)

# ============================
# CARGA DE DATOS
# ============================

st.sidebar.header("📂 Subir datasets")

file_all_seasons = st.sidebar.file_uploader("all_seasons.csv")
file_player = st.sidebar.file_uploader("player.csv")
file_team = st.sidebar.file_uploader("team.csv")
file_line_score = st.sidebar.file_uploader("line_score.csv")
file_game = st.sidebar.file_uploader("game.csv")
file_vara = st.sidebar.file_uploader("nba_puntaje_vara.csv")

if not all([file_all_seasons, file_player, file_team, file_line_score, file_game, file_vara]):
    st.warning("⛔ Cargá los 6 archivos para comenzar.")
    st.stop()

all_seasons = pd.read_csv(file_all_seasons)
player = pd.read_csv(file_player)
team = pd.read_csv(file_team)
line_score = pd.read_csv(file_line_score)
game = pd.read_csv(file_game)
df_nba = pd.read_csv(file_vara)

st.success("✔️ Datos cargados correctamente")

# ============================
# EDA INTERACTIVO
# ============================

st.header("📊 Exploración de Datos (EDA)")

col1, col2 = st.columns(2)

numeric_cols = df_nba.select_dtypes(include=['number']).columns
categorical_cols = df_nba.select_dtypes(exclude=['number']).columns

with col1:
    graph_type = st.selectbox(
        "Tipo de gráfico",
        ["Histograma", "Boxplot", "Scatterplot", "Heatmap"]
    )

with col2:
    if graph_type == "Histograma":
        col_x = st.selectbox("Variable numerica:", numeric_cols)
    elif graph_type == "Boxplot":
        col_x = st.selectbox("Variable categórica:", categorical_cols)
        col_y = st.selectbox("Variable numérica:", numeric_cols)
    else:
        col_x = st.selectbox("Variable X:", df_nba.columns)
        col_y = st.selectbox("Variable Y:", df_nba.columns)

# GENERADOR DE GRÁFICOS
if graph_type == "Histograma":
    fig, ax = plt.subplots()
    sns.histplot(df_nba[col_x], kde=True, color=COLOR_ACCENT)
    st.pyplot(fig)

elif graph_type == "Boxplot":
    fig, ax = plt.subplots()
    sns.boxplot(data=df_nba, x=col_x, y=col_y, color=COLOR_1)
    st.pyplot(fig)

elif graph_type == "Scatterplot":
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_nba, x=col_x, y=col_y, color=COLOR_1)
    st.pyplot(fig)

elif graph_type == "Heatmap":
    fig, ax = plt.subplots()
    sns.heatmap(df_nba.corr(), annot=False, cmap="viridis")
    st.pyplot(fig)

# ============================
# CLUSTERING KMEANS
# ============================

st.header("🎯 Clustering de Jugadores (Rendimiento)")

features = [
    'ts_pct_score','usg_pct_score','ast_pct_score',
    'oreb_pct_score','dreb_pct_score','net_rating_score'
]

X = df_nba[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4, random_state=42)
df_nba["cluster"] = kmeans.fit_predict(X_scaled)

st.write("### Centros de cada cluster")
st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=features))

# INTERPRETACIÓN AUTOMÁTICA
st.write("### Interpretación automática de clusters:")

summary = df_nba.groupby("cluster")[features].mean()

dominant_feature = summary.idxmax(axis=1)

interpretation_map = {
    "ts_pct_score": "Bajo uso, alta eficiencia",
    "dreb_pct_score": "Alto rebote defensivo",
    "ast_pct_score": "Creador de juego (AST)",
    "usg_pct_score": "Anotador de alto volumen"
}

df_nba["cluster_label"] = dominant_feature.map(interpretation_map)

st.dataframe(df_nba[["player_name", "cluster", "cluster_label"]].head())

# ============================
# REGRESIÓN / PREDICCIÓN DE RENDIMIENTO
# ============================

st.header("📈 Modelo Predictivo – Regresión del Rendimiento")

target = "global_score"
features_reg = [
    "ts_pct_score","usg_pct_score","dreb_pct_score","ast_pct_score",
    "oreb_pct_score","age","player_height","player_weight"
]

X = df_nba[features_reg]
y = df_nba[target]

model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)

mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.write(f"**MAE:** {mae}")
st.write(f"**R²:** {r2}")

coef_df = pd.DataFrame({
    "Variable": features_reg,
    "Importancia": model.coef_
}).sort_values("Importancia", ascending=False)

st.write("### Importancia de variables")
st.dataframe(coef_df)

# ============================
# TOP 10 JUGADORES
# ============================

st.header("🏆 Top 10 jugadores por rendimiento global")

top10 = (
    df_nba.sort_values("global_score", ascending=False)
          .head(10)[["player_name","team_abbreviation","global_score"]]
)

st.table(top10)

fig, ax = plt.subplots(figsize=(8,5))
sns.barplot(data=top10, y="player_name", x="global_score", palette="viridis")
plt.title("Top 10 jugadores (Global Score)")
st.pyplot(fig)

# ============================
# DESCARGA DE RESULTADOS
# ============================

st.download_button(
    "📥 Descargar Dataset Modificado",
    df_nba.to_csv(index=False),
    "nba_processed.csv",
    "text/csv"
)
