import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# ======================================================
# CONFIG GENERAL (COLORES + PAGE CONFIG + CSS + LOGO)
# ======================================================

st.set_page_config(page_title="NBA Analytics", layout="wide", page_icon="🏀")

COLOR_BG = "#012E40"
COLOR_ACCENT = "#F28705"
COLOR_1 = "#025159"
COLOR_2 = "#038C8C"
COLOR_3 = "#03A696"

# LOGO (arriba a la derecha)
LOGO_URL = "https://raw.githubusercontent.com/MelRossi/App-rendimiento-nba/main/Image_logo.png"

st.markdown(
    f"""
    <div style="display:flex; justify-content:flex-end; margin-top:-35px; margin-bottom:-10px;">
        <img src="{LOGO_URL}" width="130">
    </div>
    """,
    unsafe_allow_html=True
)

# CSS general
st.markdown(
    f"""
    <style>
    [data-testid="stSidebar"] {{
        min-width: 390px;
        max-width: 390px;
    }}
    .stApp {{
        background-color: {COLOR_BG};
        color: white;
    }}
    .title {{
        font-size: 45px;
        text-align: center;
        color: {COLOR_ACCENT};
        font-weight: bold;
    }}
    .subtitle {{
        text-align: center;
        color: {COLOR_3};
        margin-top: -15px;
        margin-bottom: 10px;
    }}
    .plot-container {{
        transform: scale(0.87);
        transform-origin: top left;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ======================================================
# 1. CARGA DE DATOS (pantalla principal)
# ======================================================

st.markdown("<div class='title'>🏀 NBA Performance Analytics</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>EDA, Clustering y Modelos de Predicción</div>", unsafe_allow_html=True)
st.markdown("---")

RAW = "https://raw.githubusercontent.com/MelRossi/App-rendimiento-nba/main/"

CSV_FILES = {
    "puntaje": "nba_puntaje_vara.csv",
    "all_seasons": "all_seasons_filtrado.csv",
    "player": "player_filtrado.csv",
    "team": "team_filtrado.csv",
    "game": "game_filtrado.csv",
    "line_score": "line_score_filtrado.csv",
    "resumen": "resumen.csv"
}

# ---- CARGA DE LOS 7 CSV ----
@st.cache_data
def load_default_csvs():
    dfs = {}
    for k, fname in CSV_FILES.items():
        try:
            dfs[k] = pd.read_csv(RAW + fname)
        except:
            dfs[k] = pd.DataFrame()
    return dfs

dfs_default = load_default_csvs()

# ---- CARGA DE ARCHIVO SUBIDO ----
uploaded = st.file_uploader("📂 Subir archivo CSV (opcional) para reemplazar dataset principal")

if uploaded:
    df_uploaded = pd.read_csv(uploaded)
    df_base = df_uploaded.copy()
    st.success(f"Archivo cargado correctamente: {uploaded.name}")
else:
    df_base = None
    st.info("Archivos por defecto: cargados correctamente.")

st.markdown("---")

# ======================================================
# 2. SELECCIÓN DEL DATASET PRINCIPAL
# ======================================================

st.subheader("📌 Seleccioná el dataset con el que querés trabajar")

if uploaded:
    opciones = {"Archivo subido": "uploaded"}
else:
    opciones = {
        "nba_puntaje_vara.csv (Global Score)": "puntaje",
        "all_seasons_filtrado.csv": "all_seasons",
        "player_filtrado.csv": "player",
        "team_filtrado.csv": "team",
        "game_filtrado.csv": "game",
        "line_score_filtrado.csv": "line_score",
        "resumen.csv": "resumen"
    }

choice = st.selectbox("Dataset principal:", list(opciones.keys()))

if choice == "Archivo subido":
    df_nba = df_base.copy()
else:
    df_nba = dfs_default[opciones[choice]].copy()

st.success(f"📊 Dataset seleccionado: {choice} — {df_nba.shape[0]:,} filas")

st.markdown("---")

# ======================================================
# 3. PREPARAR ETIQUETA DE POTENCIAL + ENTRENAR CLASIFICADOR
# ======================================================

TARGET = "global_score"
FEATURES = [
    "ts_pct_score","usg_pct_score","dreb_pct_score","ast_pct_score",
    "oreb_pct_score","age","player_height","player_weight"
]

potential_threshold = None
classifier = None
scaler_clf = None
clf_acc = None

if TARGET in df_nba.columns:
    try:
        potential_threshold = df_nba[TARGET].dropna().quantile(0.75)
        df_nba["potencial_bin"] = (df_nba[TARGET] >= potential_threshold).astype(int)

        df_clf = df_nba[FEATURES + ["potencial_bin"]].dropna()
        if df_clf.shape[0] >= 40:
            scaler_clf = StandardScaler()
            Xs = scaler_clf.fit_transform(df_clf[FEATURES])
            y = df_clf["potencial_bin"]
            Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)
            classifier = LogisticRegression(max_iter=500)
            classifier.fit(Xtr, ytr)
            clf_acc = accuracy_score(yte, classifier.predict(Xte))
    except:
        pass

# ======================================================
# 4. SIDEBAR — PREDICCIÓN
# ======================================================

st.sidebar.markdown(f"<h3 style='color:{COLOR_ACCENT};'>🎯 Predicción de Potencial</h3>", unsafe_allow_html=True)

DEFAULT_R = {
    "ts_pct_score": (0.0, 1.0, 0.55),
    "usg_pct_score": (0.0, 1.0, 0.25),
    "dreb_pct_score": (0.0, 0.4, 0.12),
    "ast_pct_score": (0.0, 1.0, 0.12),
    "oreb_pct_score": (0.0, 0.4, 0.06),
    "age": (16, 45, 26),
    "player_height": (160, 230, 198),
    "player_weight": (60, 140, 95)
}

input_vals = {}
for col in FEATURES:
    vmin, vmax, vmean = DEFAULT_R[col]
    if col in df_nba.columns:
        try:
            vmin = float(df_nba[col].min())
            vmax = float(df_nba[col].max())
            vmean = float(df_nba[col].median())
        except:
            pass

    if col in ["age","player_height","player_weight"]:
        input_vals[col] = st.sidebar.slider(col, int(vmin), int(vmax), int(vmean))
    else:
        input_vals[col] = st.sidebar.slider(col, float(vmin), float(vmax), float(vmean), step=0.01)

if st.sidebar.button("🔮 Predecir"):
    user_df = pd.DataFrame([input_vals])
    method = ""
    if classifier is not None:
        p = classifier.predict(scaler_clf.transform(user_df))[0]
        label = "Tiene potencial" if p == 1 else "No tiene potencial"
        method = "Clasificador (LogReg)"
    else:
        ts = user_df["ts_pct_score"].iloc[0]
        usg = user_df["usg_pct_score"].iloc[0]
        label = "Tiene potencial" if (ts > 0.58 and usg > 0.20) else "No tiene potencial"
        method = "Heurística simple"

    st.sidebar.success(f"Resultado: {label} ({method})")

# ======================================================
# 5. EDA — BARRA / LÍNEA / SCATTER
# ======================================================

st.markdown(
    f"<h2 style='color:{COLOR_ACCENT}; font-weight:700;'>📊 Exploración de Datos (EDA)</h2>",
    unsafe_allow_html=True
)

num_cols = df_nba.select_dtypes(include=['number']).columns.tolist()
all_cols = df_nba.columns.tolist()

c1,c2,c3 = st.columns(3)

with c1:
    tipo = st.selectbox("Tipo de gráfico", ["Barra","Línea","Scatterplot"])

with c2:
    xcol = st.selectbox("Eje X:", all_cols)

with c3:
    ycol = st.selectbox("Eje Y:", num_cols)

fig, ax = plt.subplots(figsize=(8,5))
fig.patch.set_facecolor(COLOR_BG)
ax.set_facecolor(COLOR_BG)

plt.rcParams.update({"text.color":"white","axes.labelcolor":"white","xtick.color":"white","ytick.color":"white"})

try:
    if tipo == "Barra":
        g = df_nba.groupby(xcol)[ycol].mean().reset_index()
        sns.barplot(data=g, x=xcol, y=ycol, ax=ax, palette="crest")
        ax.set_title(f"{ycol} por {xcol}", color=COLOR_3)

    elif tipo == "Línea":
        g = df_nba[[xcol,ycol]].dropna().sort_values(xcol)
        sns.lineplot(data=g, x=xcol, y=ycol, ax=ax, color=COLOR_2)
        ax.set_title(f"{ycol} vs {xcol}", color=COLOR_3)

    else:
        sns.scatterplot(data=df_nba, x=xcol, y=ycol, ax=ax, color=COLOR_ACCENT)
        ax.set_title(f"Scatter: {ycol} vs {xcol}", color=COLOR_3)

except:
    st.warning("No se pudo generar gráfico con esas columnas.")

st.markdown('<div class="plot-container">', unsafe_allow_html=True)
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")

# ======================================================
# 6. TOP 10 GLOBAL SCORE (SIEMPRE SOBRE nba_puntaje_vara.csv)
# ======================================================

st.markdown(
    f"<h2 style='color:{COLOR_ACCENT}; font-weight:700;'>🏆 Top 10 jugadores por Global Score</h2>",
    unsafe_allow_html=True
)

df_model = dfs_default["puntaje"].copy()  # <-- SIEMPRE ESTE

if not df_model.empty and "global_score" in df_model.columns:
    try:
        top10 = (
            df_model.sort_values("global_score", ascending=False)
                    .head(10)[["player_name","team_abbreviation","global_score"]]
        )
        st.table(top10)

        fig, ax = plt.subplots(figsize=(8,5))
        fig.patch.set_facecolor(COLOR_BG)
        ax.set_facecolor(COLOR_BG)

        sns.barplot(data=top10, y="player_name", x="global_score", palette="crest", ax=ax)
        ax.set_title("Top 10 jugadores (Global Score)", color=COLOR_3)
        ax.tick_params(colors='white')

        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error generando Top 10: {e}")
else:
    st.info("El archivo de puntaje no contiene 'global_score'.")

st.markdown("---")

# ======================================================
# 7. RANDOM FOREST (SIEMPRE SOBRE nba_puntaje_vara.csv)
# ======================================================

st.markdown(
    f"<h2 style='color:{COLOR_ACCENT}; font-weight:700;'>📈 Modelo Predictivo — RandomForestRegressor</h2>",
    unsafe_allow_html=True
)

req = [
    'age','player_height','player_weight',
    'oreb_pct_score','dreb_pct_score','usg_pct_score',
    'ts_pct_score','ast_pct_score'
]

if all([c in df_model.columns for c in req + ["global_score"]]):
    df_rf = df_model.dropna(subset=req + ["global_score"])
    if df_rf.shape[0] >= 30:

        X = df_rf[req]
        y = df_rf["global_score"]

        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)

        rf = RandomForestRegressor(n_estimators=300, random_state=42)
        rf.fit(Xtr, ytr)

        ypred = rf.predict(Xte)

        colA, colB = st.columns(2)
        colA.metric("MAE", f"{mean_absolute_error(yte, ypred):.4f}")
        colB.metric("R²", f"{r2_score(yte, ypred):.4f}")

        importancia = pd.Series(rf.feature_importances_, index=req).sort_values()

        st.write("### Importancia de variables")
        fig, ax = plt.subplots(figsize=(7,5))
        fig.patch.set_facecolor(COLOR_BG)
        ax.set_facecolor(COLOR_BG)
        importancia.plot(kind='barh', ax=ax, color=COLOR_ACCENT)
        ax.tick_params(colors='white')

        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    else:
        st.info("El dataset de puntaje no tiene suficientes filas limpias para entrenar RandomForest.")
else:
    st.info("El archivo de puntaje no contiene todas las columnas necesarias.")

st.markdown("---")

# ======================================================
# 8. KMEANS (SIEMPRE SOBRE nba_puntaje_vara.csv)
# ======================================================

st.markdown(
    f"<h2 style='color:{COLOR_ACCENT}; font-weight:700;'>🎯 Clustering KMeans (6 features)</h2>",
    unsafe_allow_html=True
)

cluster_cols = [
    'ts_pct_score','usg_pct_score','ast_pct_score',
    'oreb_pct_score','dreb_pct_score','net_rating_score'
]

if all([c in df_model.columns for c in cluster_cols]):
    dfc = df_model.dropna(subset=cluster_cols)
    if dfc.shape[0] >= 20:
        scaler = StandardScaler()
        Xsc = scaler.fit_transform(dfc[cluster_cols])

        kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
        dfc["cluster"] = kmeans.fit_predict(Xsc)

        st.write("### Centros del cluster (estandarizados)")
        st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=cluster_cols))

        summary = dfc.groupby("cluster")[cluster_cols].mean()
        dominant = summary.idxmax(axis=1)

        interpretacion = {
            'ts_pct_score': 'Bajo uso, alta eficiencia',
            'dreb_pct_score': 'Alto rebote defensivo',
            'ast_pct_score': 'Creador de juego',
            'usg_pct_score': 'Anotador de alto volumen',
            'net_rating_score': 'Impacto neto positivo',
            'oreb_pct_score': 'Rebote ofensivo'
        }

        labels = dominant.map(interpretacion).fillna("—")
        st.write("### Etiquetas interpretativas")
        st.table(labels)

    else:
        st.info("No hay suficientes filas en nba_puntaje_vara.csv para clustering.")
else:
    st.info("El archivo de puntaje no contiene las columnas necesarias para clustering.")

st.markdown("---")

# ======================================================
# 9. DESCARGA FINAL
# ======================================================

st.markdown(
    f"<h2 style='color:{COLOR_ACCENT}; font-weight:700;'>💾 Descargar dataset</h2>",
    unsafe_allow_html=True
)

st.download_button(
    "Descargar CSV",
    df_nba.to_csv(index=False).encode("utf-8"),
    "dataset_procesado.csv",
    "text/csv"
)
