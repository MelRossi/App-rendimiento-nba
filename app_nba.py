# app_final.py
"""
App Streamlit - NBA Performance Analytics (Multi CSV)
Carga automática desde tu repo GitHub, EDA (Bar / Line / Scatter),
RandomForest Regression, KMeans clustering y Predicción manual de 'potencial'.

Copia/pega este archivo y ejecútalo con:
    streamlit run app_final.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, accuracy_score

# ----------------------
# Configuración inicial
# ----------------------
st.set_page_config(page_title="NBA Performance Analytics", layout="wide", page_icon="🏀")

# Paleta (de tu app original)
COLOR_BG = "#012E40"
COLOR_ACCENT = "#F28705"
COLOR_1 = "#025159"
COLOR_2 = "#038C8C"
COLOR_3 = "#03A696"

# Ensanchar sidebar y escala de plots (CSS)
st.markdown(
    f"""
    <style>
    /* Sidebar ancho */
    [data-testid="stSidebar"] {{
        min-width: 400px;
        max-width: 400px;
    }}
    /* Fondo app */
    .stApp {{
        background-color: {COLOR_BG};
        color: white;
    }}
    /* Título */
    .title {{
        font-size: 44px;
        text-align: center;
        color: {COLOR_ACCENT};
        font-weight: 700;
    }}
    .subtitle {{
        color: {COLOR_3};
        text-align:center;
        margin-top: -10px;
        margin-bottom: 18px;
    }}
    /* Reduce visualmente tamaño de plots (no altera resolución) */
    .plot-container {{
        transform: scale(0.88);
        transform-origin: top left;
    }}
    /* Ajuste estético para tablas */
    .stDataFrame table {{
        color: white;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# ----------------------
# Archivos CSV en GitHub (raw)
# ----------------------
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

@st.cache_data
def load_all_csvs(base_url, csv_files):
    dfs = {}
    msgs = {}
    for key, fname in csv_files.items():
        url = base_url + fname
        try:
            df = pd.read_csv(url)
            dfs[key] = df
            msgs[key] = f"Cargado: {fname} ({len(df):,} filas)"
        except Exception as e:
            dfs[key] = pd.DataFrame()
            msgs[key] = f"No se pudo cargar {fname}: {e}"
    return dfs, msgs

dfs, load_msgs = load_all_csvs(RAW_BASE, CSV_FILES)

# ----------------------
# Encabezado y estado
# ----------------------
st.markdown("<div class='title'>🏀 NBA Performance Analytics (Multi CSV)</div>", unsafe_allow_html=True)
st.markdown(f"<div class='subtitle'>EDA, Clustering y Predicción simple de potencial</div>", unsafe_allow_html=True)
st.markdown("---")

col_left, col_right = st.columns([3,1])
with col_left:
    st.header("📂 Datasets cargados")
    for k in CSV_FILES.keys():
        m = load_msgs.get(k, "")
        if m.startswith("Cargado"):
            st.success(m)
        else:
            st.warning(m)
with col_right:
    st.write("")  # espacio a la derecha (no hay logo local por seguridad)

# ----------------------
# Seleccionar dataset principal
# ----------------------
# Prioridad: 'puntaje' (nba_puntaje_vara.csv) -> 'all_seasons'
df_nba = dfs.get("puntaje", pd.DataFrame()).copy()
if df_nba.empty:
    df_nba = dfs.get("all_seasons", pd.DataFrame()).copy()

if df_nba.empty:
    st.error("No se pudo cargar el dataset principal (nba_puntaje_vara.csv o all_seasons_filtrado.csv). Algunas funcionalidades estarán limitadas.")
else:
    st.success(f"Dataset principal listo: {df_nba.shape[0]:,} filas x {df_nba.shape[1]:,} columnas")

st.markdown("---")

# ----------------------
# Preparamos etiqueta 'potencial' usando percentil 75 (si existe global_score)
# ----------------------
TARGET = "global_score"
FEATURES_REG = [
    "ts_pct_score","usg_pct_score","dreb_pct_score","ast_pct_score",
    "oreb_pct_score","age","player_height","player_weight"
]

potential_threshold = None
if TARGET in df_nba.columns:
    try:
        potential_threshold = float(df_nba[TARGET].dropna().quantile(0.75))
        df_nba["potencial_bin"] = (df_nba[TARGET] >= potential_threshold).astype(int)
    except Exception:
        potential_threshold = None

# ----------------------
# Entrenamos un clasificador simple (Logistic) para usar en la sidebar si hay suficiente data
# ----------------------
classifier = None
scaler_clf = None
clf_acc = None

def train_classifier_if_possible(df, features, target_col="potencial_bin"):
    if df is None or df.empty:
        return None, None, None
    if target_col not in df.columns or not all([f in df.columns for f in features]):
        return None, None, None
    dfc = df[features + [target_col]].dropna()
    if dfc.shape[0] < 40:
        return None, None, None
    X = dfc[features]
    y = dfc[target_col]
    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42, stratify=y)
    clf = LogisticRegression(max_iter=500)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    acc = accuracy_score(yte, ypred)
    return clf, sc, acc

classifier, scaler_clf, clf_acc = train_classifier_if_possible(df_nba, FEATURES_REG, "potencial_bin")

# Mostrar estado del clasificador
st.markdown("### 🔧 Estado del clasificador de 'potencial'")
if potential_threshold is None:
    st.info("No se encontró 'global_score' para calcular el umbral de potencial.")
else:
    st.write(f"Umbral de 'potencial' (percentil 75) = **{potential_threshold:.4f}**")
if classifier is None:
    st.warning("No se entrenó un clasificador (datos insuficientes o faltan columnas). Se usará regresión o heurística para la predicción manual.")
else:
    st.success(f"Clasificador disponible — accuracy en validación ≈ {clf_acc:.3f}")

st.markdown("---")

# ----------------------
# SIDEBAR: sliders manuales + predicción de potencial
# ----------------------
st.sidebar.markdown(f"<h3 style='color:{COLOR_ACCENT};'>🚀 Predicción de Potencial (Manual)</h3>", unsafe_allow_html=True)
st.sidebar.write("Introduce valores numéricos y presiona 'Predecir'.")

# Rango dinámico o por defecto
DEFAULT_RANGES = {
    "ts_pct_score": {"min": 0.0, "max": 1.0, "mean": 0.55},
    "usg_pct_score": {"min": 0.0, "max": 1.0, "mean": 0.25},
    "dreb_pct_score": {"min": 0.0, "max": 0.4, "mean": 0.12},
    "ast_pct_score": {"min": 0.0, "max": 1.0, "mean": 0.12},
    "oreb_pct_score": {"min": 0.0, "max": 0.4, "mean": 0.06},
    "age": {"min": 16, "max": 45, "mean": 26},
    "player_height": {"min": 160, "max": 230, "mean": 198},
    "player_weight": {"min": 60, "max": 140, "mean": 95}
}

rangos = {}
for col in FEATURES_REG:
    if (not df_nba.empty) and (col in df_nba.columns):
        try:
            rangos[col] = {
                "min": float(df_nba[col].min()),
                "max": float(df_nba[col].max()),
                "mean": float(df_nba[col].median())
            }
        except Exception:
            rangos[col] = DEFAULT_RANGES[col]
    else:
        rangos[col] = DEFAULT_RANGES[col]

# Crear sliders
input_vals = {}
for col in FEATURES_REG:
    r = rangos[col]
    if 'score' in col or 'pct' in col:
        val = st.sidebar.slider(col, float(r["min"]), float(r["max"]), float(r["mean"]), step=0.01, format="%.2f")
    else:
        val = st.sidebar.slider(col, int(np.floor(r["min"])), int(np.ceil(r["max"])), int(np.round(r["mean"])), step=1)
    input_vals[col] = val

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🎯 Predecir Potencial")

if predict_button:
    df_user = pd.DataFrame([input_vals])
    result_label = None
    method = None

    # 1) Intentar clasificador entrenado
    if classifier is not None and scaler_clf is not None:
        try:
            Xusr = scaler_clf.transform(df_user[FEATURES_REG])
            p = classifier.predict(Xusr)[0]
            result_label = "Tiene potencial" if int(p) == 1 else "No tiene potencial"
            method = "Clasificador (LogisticRegression)"
        except Exception:
            result_label = None

    # 2) Si no hay clasificador, usar regresión RandomForest entrenada en sección ML (o entrenar aquí rápido)
    if result_label is None and (TARGET in df_nba.columns) and all([c in df_nba.columns for c in FEATURES_REG]):
        try:
            df_reg = df_nba.dropna(subset=FEATURES_REG + [TARGET])
            if not df_reg.empty and df_reg.shape[0] >= 30:
                Xr = df_reg[FEATURES_REG]
                yr = df_reg[TARGET]
                reg = RandomForestRegressor(n_estimators=200, random_state=42)
                reg.fit(Xr, yr)
                yhat = reg.predict(df_user[FEATURES_REG])[0]
                if potential_threshold is not None:
                    result_label = "Tiene potencial" if yhat >= potential_threshold else "No tiene potencial"
                    method = "RandomForestRegressor -> umbral (percentil 75)"
                else:
                    med = df_reg[TARGET].median()
                    result_label = "Tiene potencial" if yhat >= med else "No tiene potencial"
                    method = "RandomForestRegressor -> umbral (mediana)"
        except Exception:
            result_label = None

    # 3) Heurística simple fallback
    if result_label is None:
        ts = input_vals.get("ts_pct_score", 0)
        usg = input_vals.get("usg_pct_score", 0)
        if ts >= 0.58 and usg >= 0.20:
            result_label = "Tiene potencial"
        else:
            result_label = "No tiene potencial"
        method = "Heurística simple"

    # Mostrar resultado bonito
    st.sidebar.markdown(
        f"""
        <div style='background:{COLOR_1}; padding:10px; border-radius:8px; text-align:center;'>
            <p style='color:white; margin:0;'>Resultado</p>
            <h2 style='color:{COLOR_ACCENT}; margin:0;'>{result_label}</h2>
            <small style='color:white;'>Método: {method}</small>
        </div>
        """,
        unsafe_allow_html=True
    )

st.sidebar.markdown("---")
st.sidebar.write("Umbral (percentil 75) para 'potencial':")
st.sidebar.write(f"**{potential_threshold:.4f}**" if potential_threshold is not None else "N/A")

# ----------------------
# EDA: Barra / Línea / Scatter
# ----------------------
st.header("📊 Exploración de Datos (EDA) — Barra / Línea / Scatter")

numeric_cols = df_nba.select_dtypes(include=['number']).columns.tolist() if not df_nba.empty else []
all_cols = df_nba.columns.tolist() if not df_nba.empty else []

col1, col2, col3 = st.columns([1,1,1])
with col1:
    chart_type = st.selectbox("Tipo de gráfico:", ["Barra", "Línea", "Scatterplot"])
with col2:
    if chart_type == "Barra":
        x_col = st.selectbox("Eje X (categórico/agrupación):", all_cols)
        y_col = st.selectbox("Eje Y (numérica):", numeric_cols)
    else:
        x_col = st.selectbox("Eje X (numérica):", numeric_cols)
        y_col = st.selectbox("Eje Y (numérica):", numeric_cols, index=1 if len(numeric_cols) > 1 else 0)
with col3:
    groupby_col = st.selectbox("Color / Agrupar por (opcional):", [None] + all_cols, index=0)

# Generación del gráfico
if x_col and y_col:
    try:
        fig, ax = plt.subplots(figsize=(9,5))
        fig.patch.set_facecolor(COLOR_BG)
        ax.set_facecolor(COLOR_BG)
        plt.rcParams.update({'text.color':'white','axes.labelcolor':'white','xtick.color':'white','ytick.color':'white'})

        if chart_type == "Barra":
            if x_col in df_nba.columns and y_col in df_nba.columns:
                grouping = df_nba.groupby(x_col)[y_col].mean().reset_index()
                sns.barplot(data=grouping, x=x_col, y=y_col, ax=ax, palette="crest")
                ax.set_title(f"Media de {y_col} por {x_col}", color=COLOR_3)
                plt.xticks(rotation=45, ha="right")
            else:
                st.warning("Columnas inválidas para gráfico de barras.")
        elif chart_type == "Línea":
            temp = df_nba[[x_col,y_col]].dropna().sort_values(by=x_col)
            sns.lineplot(data=temp, x=x_col, y=y_col, ax=ax)
            ax.set_title(f"{y_col} vs {x_col} (línea)", color=COLOR_3)
        else:  # Scatter
            if groupby_col and groupby_col in df_nba.columns:
                sns.scatterplot(data=df_nba.dropna(subset=[x_col,y_col]), x=x_col, y=y_col, hue=groupby_col, ax=ax, palette="viridis")
            else:
                sns.scatterplot(data=df_nba.dropna(subset=[x_col,y_col]), x=x_col, y=y_col, ax=ax, color=COLOR_ACCENT)
            ax.set_title(f"Scatter: {y_col} vs {x_col}", color=COLOR_3)

        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error generando gráfico: {e}")
else:
    st.info("Seleccioná X y Y para generar el gráfico.")

st.markdown("---")

# ----------------------
# Top 10 Jugadores (Global Score)
# ----------------------
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

        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"No se pudo generar Top 10: {e}")
else:
    st.info("Faltan columnas 'player_name' o 'global_score' para generar Top 10.")

st.markdown("---")

# ----------------------
# Machine Learning: RandomForest Regression
# ----------------------
st.header("📈 Modelo Predictivo — RandomForest Regressor (Regresión de 'global_score')")

try:
    carac = ['age', 'player_height', 'player_weight',
             'oreb_pct_score', 'dreb_pct_score', 'usg_pct_score',
             'ts_pct_score', 'ast_pct_score']

    if all([c in df_nba.columns for c in carac + [TARGET]]):
        df_rf = df_nba.dropna(subset=carac + [TARGET])
        if df_rf.shape[0] >= 30:
            X = df_rf[carac]
            y = df_rf[TARGET]

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            rf = RandomForestRegressor(n_estimators=300, random_state=42)
            rf.fit(X_train, y_train)
            y_pred = rf.predict(X_test)

            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            colA, colB = st.columns(2)
            colA.metric("MAE", f"{mae:.4f}")
            colB.metric("R²", f"{r2:.4f}")

            importancia = pd.Series(rf.feature_importances_, index=carac).sort_values()
            st.write("### Importancia de variables (RandomForest)")
            fig, ax = plt.subplots(figsize=(7,5))
            fig.patch.set_facecolor(COLOR_BG)
            ax.set_facecolor(COLOR_BG)
            importancia.plot(kind='barh', ax=ax, color=COLOR_ACCENT)
            ax.set_xlabel("Importancia")
            ax.tick_params(colors='white')

            st.markdown('<div class="plot-container">', unsafe_allow_html=True)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No hay suficientes filas sin NA para entrenar RandomForest (mínimo 30).")
    else:
        st.info("Faltan columnas necesarias para entrenar la regresión (ver lista).")
except Exception as e:
    st.error(f"Error durante el proceso de regresión: {e}")

st.markdown("---")

# ----------------------
# Clustering KMeans
# ----------------------
st.header("🎯 Clustering de Jugadores — KMeans (6 features)")

try:
    cols_cluster = ['ts_pct_score','usg_pct_score','ast_pct_score',
                    'oreb_pct_score','dreb_pct_score','net_rating_score']
    if all([c in df_nba.columns for c in cols_cluster]):
        df_cluster = df_nba.dropna(subset=cols_cluster).copy()
        if df_cluster.shape[0] >= 10:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_cluster[cols_cluster])

            kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
            df_cluster['cluster'] = kmeans.fit_predict(X_scaled)
            # Merge cluster back to df_nba where indices align
            df_nba.loc[df_cluster.index, 'cluster'] = df_cluster['cluster']

            st.write("### Centros de cada cluster (valores estandarizados)")
            centers_df = pd.DataFrame(kmeans.cluster_centers_, columns=cols_cluster)
            st.dataframe(centers_df)

            cluster_summary = df_cluster.groupby('cluster')[cols_cluster].mean()
            dominantes = cluster_summary.idxmax(axis=1)

            interpretacion = {
                'ts_pct_score': 'Bajo uso, alta eficiencia',
                'dreb_pct_score': 'Alto rebote defensivo',
                'ast_pct_score': 'Creador de juego (AST)',
                'usg_pct_score': 'Anotador de alto volumen',
                'net_rating_score': 'Impacto neto positivo',
                'oreb_pct_score': 'Alto rebote ofensivo'
            }

            cluster_labels = dominantes.map(interpretacion).fillna("—")
            # Añadir etiquetas por cluster
            label_map = cluster_labels.to_dict()
            df_nba['cluster_label'] = df_nba['cluster'].map(label_map).fillna("—")

            st.write("### Ejemplo: jugadores por cluster y etiqueta interpretativa")
            show_cols = [c for c in ["player_name", "team_abbreviation", "cluster", "cluster_label"] if c in df_nba.columns]
            st.dataframe(df_nba[show_cols].head(30))
        else:
            st.info("No hay suficientes filas para clustering después de limpiar NA (mínimo 10).")
    else:
        st.info("Faltan columnas necesarias para clustering KMeans.")
except Exception as e:
    st.error(f"Error en clustering: {e}")

st.markdown("---")

# ----------------------
# Descarga dataset (opcional con clusters)
# ----------------------
st.header("💾 Descargar dataset procesado")
try:
    if 'cluster_label' in df_nba.columns:
        df_download = df_nba.copy()
    else:
        df_download = df_nba.copy()
    st.download_button("📥 Descargar dataset principal (CSV)", df_download.to_csv(index=False).encode('utf-8'), "nba_processed.csv", "text/csv")
except Exception as e:
    st.error(f"Error preparando el archivo para descargar: {e}")

st.caption("App generada desde: https://github.com/MelRossi/App-rendimiento-nba")
