# app_nba_full.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ============================
# CONFIG
# ============================

st.set_page_config(page_title="NBA Analytics - Multi CSV", layout="wide", page_icon="🏀")

# Colores (tomados de tu app original)
COLOR_BG = "#012E40"
COLOR_ACCENT = "#F28705"  # Naranja brillante
COLOR_1 = "#025159"    # Azul oscuro verdoso
COLOR_2 = "#038C8C"    # Cian
COLOR_3 = "#03A696"    # Verde agua

# Raw base URL del repo que nos diste
RAW_BASE = "https://raw.githubusercontent.com/MelRossi/App-rendimiento-nba/main/"

CSV_FILES = {
    "all_seasons": "all_seasons_filtrado.csv",
    "game": "game_filtrado.csv",
    "player": "player_filtrado.csv",
    "team": "team_filtrado.csv",
    "line_score": "line_score_filtrado.csv",
    "puntaje": "nba_puntaje_vara.csv",
    "resumen": "resumen.csv"
}

# Features (mantengo las tuyas)
TARGET = "global_score"
FEATURES_REG = [
    "ts_pct_score","usg_pct_score","dreb_pct_score","ast_pct_score",
    "oreb_pct_score","age","player_height","player_weight"
]
FEATURES_CLUSTER = [
    'ts_pct_score','usg_pct_score','ast_pct_score',
    'oreb_pct_score','dreb_pct_score','net_rating_score'
]

# CSS básico para fondo y títulos
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {COLOR_BG}; color: white; }}
    .title {{ font-size: 48px; text-align:center; color: {COLOR_ACCENT}; font-weight:700; }}
    .sub {{ color: {COLOR_3}; text-align:center; }}
    .sidebar .stSlider > label {{ color: white; }}
    .card {{ background-color: {COLOR_1}; padding: 10px; border-radius: 8px; }}
    </style>
    """,
    unsafe_allow_html=True
)

# ============================
# LOAD ALL CSVs (from GitHub raw)
# ============================
@st.cache_data
def load_all_csvs(base_url, csv_files):
    dfs = {}
    messages = []
    for key, fname in csv_files.items():
        url = base_url + fname
        try:
            df = pd.read_csv(url)
            dfs[key] = df
            messages.append(f"Cargado: {fname} ({len(df):,} filas)")
        except Exception as e:
            dfs[key] = pd.DataFrame()
            messages.append(f"No se pudo cargar {fname}: {e}")
    return dfs, messages

dfs, load_msgs = load_all_csvs(RAW_BASE, CSV_FILES)

# Mostrar título y estado de carga
st.markdown("<div class='title'>🏀 NBA Performance Analytics (Multi CSV)</div>", unsafe_allow_html=True)
st.markdown(f"<div class='sub'>EDA, Clustering y Predicción simple de potencial</div>", unsafe_allow_html=True)
st.markdown("---")

col1, col2 = st.columns([3,1])
with col1:
    st.header("📂 Datasets cargados")
    for m in load_msgs:
        if m.startswith("Cargado"):
            st.success(m)
        else:
            st.warning(m)
with col2:
    st.image("Image_logo.png" if 'Image_logo.png' in st.file_uploader else None, width=100)  # si existe en local del repo

# Referencia principal: dataframe de puntaje (tu 'nba_puntaje_vara.csv')
df_nba = dfs.get("puntaje", pd.DataFrame())

# Si está vacío, intentar tomar otro (player/resumen) como fallback
if df_nba.empty:
    df_nba = dfs.get("all_seasons", pd.DataFrame())

# Si sigue vacío, mostramos aviso y paramos el EDA interactivo (pero permitimos ver archivos cargados)
if df_nba.empty:
    st.error("No se encontró el dataset principal para análisis (nba_puntaje_vara.csv o all_seasons_filtrado.csv). Algunas funcionalidades estarán limitadas.")
else:
    st.success(f"Dataset principal listo: {df_nba.shape[0]:,} filas x {df_nba.shape[1]:,} columnas")

st.markdown("---")

# ============================
# PREP: crear etiqueta binaria 'potencial' con percentil 75
# ============================
classifier = None
scaler = None
potential_threshold = None

def prepare_potential_label(df, target_col="global_score", q=0.75):
    if target_col not in df.columns:
        return df, None
    thresh = df[target_col].dropna().quantile(q)
    df = df.copy()
    df["potencial_bin"] = (df[target_col] >= thresh).astype(int)
    return df, float(thresh)

if not df_nba.empty and TARGET in df_nba.columns:
    df_nba, potential_threshold = prepare_potential_label(df_nba, TARGET, 0.75)

# Entrenar un clasificador simple si hay suficientes filas y las features existen
def train_simple_classifier(df, features, target_col="potencial_bin"):
    # verificar columnas
    if df is None or df.empty:
        return None, None
    if not all([f in df.columns for f in features]) or target_col not in df.columns:
        return None, None
    # limpiar nans
    dfc = df[features + [target_col]].dropna()
    if dfc.shape[0] < 30:
        return None, None
    X = dfc[features]
    y = dfc[target_col]
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=500)
    clf.fit(Xtr, ytr)
    # opcional: evaluar
    try:
        ypred = clf.predict(Xte)
        acc = accuracy_score(yte, ypred)
    except Exception:
        acc = None
    return (clf, sc, acc)

if potential_threshold is not None:
    clf_res = train_simple_classifier(df_nba, FEATURES_REG, "potencial_bin")
    if clf_res is not None:
        classifier, scaler, clf_acc = clf_res
    else:
        classifier, scaler, clf_acc = None, None, None
else:
    classifier, scaler, clf_acc = None, None, None

# Mostrar info del clasificador
st.markdown("### 🔧 Estado del clasificador de 'potencial'")
if potential_threshold is None:
    st.info("No hay columna 'global_score' para calcular etiqueta 'potencial'. La predicción usará heurística simple.")
else:
    st.write(f"Umbral de potencial (percentil 75) = **{potential_threshold:.4f}**")
    if classifier is not None:
        st.success(f"Clasificador entrenado con {len(df_nba.dropna(subset=FEATURES_REG)):,} filas. Accuracy (validación) ≈ {clf_acc:.3f}" if clf_acc else "Clasificador entrenado.")
    else:
        st.warning("No se entrenó un clasificador (datos insuficientes o faltan columnas). Se aplicará heurística basada en umbral/regresión.")

st.markdown("---")

# ============================
# SIDEBAR: PREDICCIÓN MANUAL (sliders numéricos)
# ============================
st.sidebar.markdown(f"<h3 style='color:{COLOR_ACCENT};'>🚀 Predicción de Potencial (Manual)</h3>", unsafe_allow_html=True)
st.sidebar.write("Introduce valores numéricos para el jugador. Luego presiona 'Predecir'.")

# Determinar rangos dinámicos a partir de df_nba o default
DEFAULT_RANGES = {
    "ts_pct_score": {"min": 0.0, "max": 1.0, "mean": 0.5},
    "usg_pct_score": {"min": 0.0, "max": 1.0, "mean": 0.25},
    "dreb_pct_score": {"min": 0.0, "max": 1.0, "mean": 0.1},
    "ast_pct_score": {"min": 0.0, "max": 1.0, "mean": 0.1},
    "oreb_pct_score": {"min": 0.0, "max": 1.0, "mean": 0.05},
    "age": {"min": 16, "max": 45, "mean": 26},
    "player_height": {"min": 160, "max": 230, "mean": 198},
    "player_weight": {"min": 60, "max": 140, "mean": 95}
}

rangos = {}
for col in FEATURES_REG:
    if not df_nba.empty and col in df_nba.columns:
        try:
            rangos[col] = {
                "min": float(df_nba[col].min()),
                "max": float(df_nba[col].max()),
                "mean": float(df_nba[col].median())
            }
        except Exception:
            rangos[col] = DEFAULT_RANGES.get(col, {"min":0,"max":1,"mean":0.5})
    else:
        rangos[col] = DEFAULT_RANGES.get(col, {"min":0,"max":1,"mean":0.5})

# Crear sliders
input_vals = {}
for col in FEATURES_REG:
    r = rangos[col]
    if 'score' in col or 'pct' in col:
        step = 0.01
        fmt = "%.2f"
        val = st.sidebar.slider(f"{col}", float(r["min"]), float(r["max"]), float(r["mean"]), step=step, format=fmt)
    else:
        step = 1
        val = st.sidebar.slider(f"{col}", int(np.floor(r["min"])), int(np.ceil(r["max"])), int(np.round(r["mean"])), step=step)
    input_vals[col] = val

st.sidebar.markdown("---")
if st.sidebar.button("🎯 Predecir Potencial"):
    # construir df usuario
    df_user = pd.DataFrame([input_vals])
    predicted_label = None
    method_used = None

    # 1) intentar usar clasificador entrenado
    if classifier is not None and scaler is not None:
        try:
            Xusr = scaler.transform(df_user[FEATURES_REG])
            pred = classifier.predict(Xusr)[0]
            predicted_label = "Tiene potencial" if int(pred) == 1 else "No tiene potencial"
            method_used = "Clasificador (Logistic)"
        except Exception as e:
            predicted_label = None

    # 2) si no hay clasificador, intentar usar regresión: entrenar regresión simple y comparar con umbral
    if predicted_label is None:
        # si existe global_score en df_nba entreno regresión simple
        if TARGET in df_nba.columns and all([c in df_nba.columns for c in FEATURES_REG]):
            try:
                df_reg = df_nba.dropna(subset=FEATURES_REG + [TARGET])
                if not df_reg.empty:
                    X = df_reg[FEATURES_REG]
                    y = df_reg[TARGET]
                    reg = LinearRegression()
                    reg.fit(X, y)
                    yhat = reg.predict(df_user[FEATURES_REG])[0]
                    if potential_threshold is not None:
                        predicted_label = "Tiene potencial" if yhat >= potential_threshold else "No tiene potencial"
                        method_used = "Regresión -> umbral (percentil 75)"
                    else:
                        # fallback heuristic: arriba de la media
                        med = df_reg[TARGET].median()
                        predicted_label = "Tiene potencial" if yhat >= med else "No tiene potencial"
                        method_used = "Regresión -> umbral (media)"
            except Exception as e:
                predicted_label = None

    # 3) heurística final: usar ts_pct_score y usg_pct_score simple rule (ejemplo)
    if predicted_label is None:
        ts = input_vals.get("ts_pct_score", 0)
        usg = input_vals.get("usg_pct_score", 0)
        # heurística simple: alta eficiencia y moderado-alto uso -> potencial
        if ts >= 0.58 and usg >= 0.2:
            predicted_label = "Tiene potencial"
        else:
            predicted_label = "No tiene potencial"
        method_used = "Heurística simple"

    # Mostrar resultado
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        f"""
        <div style='text-align:center; padding:10px; border-radius:8px; background:{COLOR_1};'>
            <p style='margin:0; color:white;'>Resultado</p>
            <h2 style='margin:0; color:{COLOR_ACCENT};'>{predicted_label}</h2>
            <small style='color:white;'>Método: {method_used}</small>
        </div>
        """,
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")
st.sidebar.write("Umbral de potencial (percentil 75) calculado a partir del dataset cargado: ")
st.sidebar.write(f"**{potential_threshold:.4f}**" if potential_threshold is not None else "N/A")

# ============================
# MAIN: EDA (Barras, Línea, Scatter)
# ============================
st.header("📊 Exploración de Datos (EDA) — Barras / Línea / Scatter")

# Opciones de columna para UI
numeric_cols = df_nba.select_dtypes(include=['number']).columns.tolist() if not df_nba.empty else []
all_cols = df_nba.columns.tolist() if not df_nba.empty else []

colA, colB, colC = st.columns([1,1,1])
with colA:
    chart_type = st.selectbox("Tipo de gráfico:", ["Barra", "Línea", "Scatterplot"])
with colB:
    if chart_type in ["Barra"]:
        x_col = st.selectbox("Eje X (categórico o agrupación):", all_cols)
        y_col = st.selectbox("Eje Y (numérica/agrupación):", numeric_cols)
    elif chart_type in ["Línea",]:
        x_col = st.selectbox("Eje X (numérica/ordenable):", numeric_cols)
        y_col = st.selectbox("Eje Y (numérica):", numeric_cols, index=0)
    else:  # Scatter
        x_col = st.selectbox("Eje X (numérica):", numeric_cols)
        y_col = st.selectbox("Eje Y (numérica):", numeric_cols, index=1 if len(numeric_cols)>1 else 0)
with colC:
    groupby_col = st.selectbox("Color / Agrupar por (opcional):", [None] + all_cols, index=0)

# Generar gráfico
if x_col and y_col:
    try:
        fig, ax = plt.subplots(figsize=(8,5))
        fig.patch.set_facecolor(COLOR_BG)
        ax.set_facecolor(COLOR_BG)
        plt.rcParams.update({'text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white'})

        if chart_type == "Barra":
            # si y_col es numérico y x_col es categórico -> promedio
            if x_col in df_nba.columns and y_col in df_nba.columns:
                grouping = df_nba.groupby(x_col)[y_col].mean().reset_index()
                sns.barplot(data=grouping, x=x_col, y=y_col, ax=ax, palette="crest")
                ax.set_title(f"Media de {y_col} por {x_col}", color=COLOR_3)
                plt.xticks(rotation=45, ha="right")
            else:
                st.warning("Columnas inválidas para gráfico de barras.")
        elif chart_type == "Línea":
            # ordenar por x_col y plot
            temp = df_nba[[x_col,y_col]].dropna().sort_values(by=x_col)
            sns.lineplot(data=temp, x=x_col, y=y_col, ax=ax)
            ax.set_title(f"{y_col} vs {x_col} (línea)", color=COLOR_3)
        else:  # Scatter
            if groupby_col and groupby_col in df_nba.columns:
                sns.scatterplot(data=df_nba.dropna(subset=[x_col,y_col]), x=x_col, y=y_col, hue=groupby_col, ax=ax, palette="viridis")
            else:
                sns.scatterplot(data=df_nba.dropna(subset=[x_col,y_col]), x=x_col, y=y_col, ax=ax, color=COLOR_ACCENT)
            ax.set_title(f"Scatter: {y_col} vs {x_col}", color=COLOR_3)

        # estilo
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Error generando gráfico: {e}")
else:
    st.info("Seleccioná X y Y para generar el gráfico.")

st.markdown("---")

# ============================
# TOP 10 Jugadores (si existen columnas)
# ============================
st.header("🏆 Top 10 jugadores por Global Score (si aplica)")
if not df_nba.empty and "player_name" in df_nba.columns and TARGET in df_nba.columns:
    try:
        top10 = df_nba.sort_values(TARGET, ascending=False).head(10)[["player_name","team_abbreviation",TARGET]].drop_duplicates()
        st.table(top10)
        fig, ax = plt.subplots(figsize=(8,5))
        fig.patch.set_facecolor(COLOR_BG)
        ax.set_facecolor(COLOR_BG)
        sns.barplot(data=top10, y="player_name", x=TARGET, ax=ax, palette="crest")
        ax.set_title("Top 10 jugadores (Global Score)", color=COLOR_3)
        ax.tick_params(colors='white')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"No se pudo generar Top 10: {e}")
else:
    st.info("Faltan columnas 'player_name' o 'global_score' para generar Top 10.")

# ============================
# DESCARGA
# ============================
st.markdown("---")
st.header("💾 Descarga")
if not df_nba.empty:
    st.download_button("📥 Descargar dataset principal (csv)", df_nba.to_csv(index=False).encode('utf-8'), "nba_principal.csv", "text/csv")
else:
    st.info("No hay dataset principal para descargar.")

st.caption("App generada a partir del repositorio: https://github.com/MelRossi/App-rendimiento-nba")

