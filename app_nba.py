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
# CONFIGURACIÓN GENERAL Y ESTILOS
# ============================

# Establece el layout en modo 'wide' para pantalla completa
st.set_page_config(
    page_title="NBA Analytics Dashboard",
    layout="wide",
    page_icon="🏀"
)

# URL para el dataset principal si no se sube un archivo (FALLBACK)
# NOTA: Debes reemplazar esta URL con la ruta real de tu archivo en GitHub.
GITHUB_MAIN_CSV_URL = "https://raw.githubusercontent.com/user/repo/main/nba_puntaje_vara.csv"

# PALETA DE COLORES PROPIA
COLOR_BG = "#012E40"
COLOR_ACCENT = "#F28705" # Naranja brillante
COLOR_1 = "#025159"    # Azul oscuro verdoso
COLOR_2 = "#038C8C"    # Cian
COLOR_3 = "#03A696"    # Verde agua

st.markdown(
    f"""
    <style>
    /* Estilos generales para la aplicación */
    .title {{
        font-size: 50px;
        text-align: center;
        font-weight: bold;
        color: {COLOR_ACCENT};
    }}
    .stApp {{
        background-color: {COLOR_BG};
    }}
    /* Estilos para los headers de las secciones */
    .stHeader, h1, h2, h3, h4, h5, h6 {{
        color: {COLOR_3} !important;
    }}
    /* Ajuste para el sidebar */
    .css-1d391kg {{
        background-color: #001f2b; /* Fondo un poco más oscuro para el sidebar */
    }}
    .sidebar-header {{
        color: {COLOR_ACCENT};
        font-size: 24px;
        font-weight: bold;
        margin-bottom: 10px;
    }}
    /* Estilo para los gráficos de Matplotlib/Seaborn dentro de Streamlit */
    .stPlot {{
        background-color: {COLOR_BG};
    }}
    /* Configurar el color de fondo de las figuras para que coincida con el fondo */
    .plt-container {{
        background-color: {COLOR_BG} !important;
    }}
    /* Ajuste para que el texto de los widgets sea blanco */
    .stSlider label, .stSelectbox label, .stDownloadButton, .stButton, .stMarkdown, .stTable, .dataframe-content, .stTextInput, .stFileUploader label, .stRadio label {{
        color: white !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# DEFINICIONES DE MODELO Y DATOS
# ============================

TARGET = "global_score"
FEATURES_REG = [
    "ts_pct_score","usg_pct_score","dreb_pct_score","ast_pct_score",
    "oreb_pct_score","age","player_height","player_weight"
]
FEATURES_CLUSTER = [
    'ts_pct_score','usg_pct_score','ast_pct_score',
    'oreb_pct_score','dreb_pct_score','net_rating_score'
]

# RANGOS PREDETERMINADOS (Fallback para los sliders)
DEFAULT_RANGES = {
    "ts_pct_score": {"min": 0.4, "max": 0.8, "mean": 0.6},
    "usg_pct_score": {"min": 0.1, "max": 0.4, "mean": 0.25},
    "dreb_pct_score": {"min": 0.05, "max": 0.35, "mean": 0.2},
    "ast_pct_score": {"min": 0.05, "max": 0.4, "mean": 0.2},
    "oreb_pct_score": {"min": 0.0, "max": 0.2, "mean": 0.1},
    "age": {"min": 18, "max": 40, "mean": 27},
    "player_height": {"min": 170, "max": 220, "mean": 200},
    "player_weight": {"min": 70, "max": 130, "mean": 100}
}

# ============================
# LÓGICA DE CARGA DE DATOS (Centralizada)
# ============================

@st.cache_data
def load_data(source, url=None):
    """Carga el dataset principal, ya sea por upload o por URL."""
    if source == "upload" and st.session_state.uploaded_file is not None:
        try:
            df = pd.read_csv(st.session_state.uploaded_file)
            return df, "Archivo subido exitosamente."
        except Exception as e:
            st.error(f"Error al cargar el archivo CSV: {e}")
            return pd.DataFrame(), f"Error al cargar el archivo: {e}"
    
    elif source == "default" and url is not None:
        try:
            # st.cache_data funciona como cache para la carga remota también
            df = pd.read_csv(url)
            return df, f"Dataset principal precargado desde GitHub: {url}"
        except Exception as e:
            st.warning(f"No se pudo precargar el dataset desde GitHub. Sube un archivo. Error: {e}")
            return pd.DataFrame(), "No se pudo precargar el dataset."
            
    return pd.DataFrame(), "Esperando carga de datos."

# Lógica de Entrenamiento del Modelo
@st.cache_resource
def train_regression_model(data, features, target_col):
    """Entrena y devuelve el modelo de Regresión Lineal y el conjunto de referencia."""
    
    if data is None or data.empty:
        return None, None
        
    required_cols = features + [target_col]
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        st.error(f"Error de columna: Faltan las siguientes columnas para el modelo de Regresión: {', '.join(missing_cols)}")
        return None, None

    try:
        data_clean = data[required_cols].dropna() 
        
        if data_clean.empty:
            st.error("Error: Después de eliminar valores nulos, el dataset para el modelo está vacío.")
            return None, None
            
        X = data_clean[features]
        y = data_clean[target_col]

        model = LinearRegression()
        model.fit(X, y)
        return model, X # X se devuelve como referencia para la evaluación
    except Exception as e:
        st.error(f"Error inesperado durante el entrenamiento del modelo: {e}")
        return None, None

# ============================
# CONTENIDO PRINCIPAL: TÍTULO Y CARGA
# ============================

st.markdown("<h1 class='title'>🏀 NBA Performance Analytics</h1>", unsafe_allow_html=True)

st.markdown(
    f"""
    <h4 style="color:{COLOR_3}; text-align:center;">
        Análisis avanzado de rendimiento y clusters
    </h4>
    """,
    unsafe_allow_html=True,
)

st.header("📂 Fuente de Datos")
col_upload, col_source_info = st.columns([2, 1])

# Inicializar estado para el uploader si no existe
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

with col_upload:
    uploaded_file = st.file_uploader(
        "Sube tu archivo 'nba_puntaje_vara.csv' para reemplazar el precargado.", 
        type=['csv'], 
        key='main_uploader'
    )
    # Actualizar el estado de sesión
    if uploaded_file is not None and st.session_state.uploaded_file != uploaded_file:
         st.session_state.uploaded_file = uploaded_file
         st.rerun()

# Determinar la fuente de datos a cargar
if st.session_state.uploaded_file:
    df_nba, load_status_message = load_data("upload")
    source_message = "Local (Archivo subido)"
else:
    df_nba, load_status_message = load_data("default", GITHUB_MAIN_CSV_URL)
    source_message = "GitHub (Precargado por defecto)"

with col_source_info:
    st.info(f"Fuente actual: **{source_message}**")
    if "Error" in load_status_message:
        st.error(load_status_message)
    else:
        st.success(load_status_message)

# Entrenar el modelo y verificar la carga
model, X_ref = train_regression_model(df_nba, FEATURES_REG, TARGET)

# ============================
# BARRA LATERAL: PREDICCIÓN MANUAL
# ============================

st.sidebar.markdown("<p class='sidebar-header'>🚀 Predicción Individual</p>", unsafe_allow_html=True)

if model is None:
    st.sidebar.warning("El modelo de predicción no está disponible. Verifica si el dataset contiene las columnas requeridas.")

datos_usuario = {}

# Determinar los rangos a usar: Default si no hay datos, o dinámicos si hay datos
rangos_prediccion = {}
for col in FEATURES_REG:
    if not df_nba.empty and col in df_nba.columns:
        rangos_prediccion[col] = {
            "min": df_nba[col].min(),
            "max": df_nba[col].max(),
            "mean": df_nba[col].mean()
        }
    else:
         rangos_prediccion[col] = DEFAULT_RANGES[col]


# Crear los Sliders en la barra lateral usando los rangos definidos
for col in FEATURES_REG:
    rango = rangos_prediccion.get(col, DEFAULT_RANGES[col]) # Usar el rango calculado o el default
    
    # Determinar el paso (step) del slider
    if 'score' in col or 'pct' in col:
        step_val = 0.01 
        fmt = "%.2f"
    elif col in ['age', 'player_height', 'player_weight']:
        step_val = 1.0 
        fmt = "%g"
    else:
        step_val = 1.0 
        fmt = "%g"
    
    # Asegurar que el valor inicial esté dentro del rango
    mean_val_safe = max(float(rango["min"]), min(float(rango["max"]), float(rango["mean"])))
    
    valor = st.sidebar.slider(
        f"**{col}**", 
        float(rango["min"]), 
        float(rango["max"]), 
        float(mean_val_safe), 
        step=step_val,
        format=fmt
    )
    datos_usuario[col] = valor

st.sidebar.markdown("---")

# Botón y Lógica de Predicción
if st.sidebar.button(":dart: **Predecir Global Score**"):
    if model is not None:
        try:
            df_usuario = pd.DataFrame([datos_usuario])
            df_usuario_final = df_usuario[FEATURES_REG]

            # Realizar la predicción
            prediccion = model.predict(df_usuario_final)[0]
            
            # Mostrar el resultado con un formato llamativo
            st.sidebar.markdown(f"""
            <div style="text-align: center; border: 2px solid {COLOR_ACCENT}; padding: 10px; border-radius: 10px; background-color: {COLOR_1};">
                <p style="font-size: 18px; margin: 0; color: white;">Global Score Estimado:</p>
                <p style="font-size: 36px; font-weight: bold; margin: 0; color: {COLOR_ACCENT};">{prediccion:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
        except Exception as e:
            st.sidebar.error(f"Error en la predicción. Detalle: {e}")
    else:
        st.sidebar.warning("Carga un archivo válido para entrenar el modelo antes de predecir.")

st.sidebar.markdown("---")

# Detener ejecución si no hay datos
if df_nba.empty:
    st.info("La interfaz de análisis avanzado (EDA, Clustering y Regresión) se activará una vez que se cargue un dataset válido.")
    st.stop()
    
# ============================
# 1. EDA INTERACTIVO (Gráficos Ajustados)
# ============================

st.header("📊 Exploración de Datos (EDA)")

col1, col2, col3 = st.columns([1, 1, 1])

numeric_cols = df_nba.select_dtypes(include=['number']).columns
all_cols = df_nba.columns

# Filtrar las opciones para incluir solo Scatterplot, Barra y Línea, más Histograma
graph_options = ["Histograma", "Gráfico de dispersión (Scatterplot)", "Gráfico de barras", "Gráfico de línea"]

with col1:
    graph_type = st.selectbox(
        "Tipo de gráfico:",
        graph_options,
        key="eda_type"
    )

with col2:
    if graph_type == "Histograma":
        col_x = st.selectbox("Variable (Numérica/Continua):", numeric_cols, key="col_hist")
    elif graph_type in ["Gráfico de dispersión (Scatterplot)", "Gráfico de línea"]:
        col_x = st.selectbox("Variable X (Numérica/Continua):", numeric_cols, key="col_x_cont")
    elif graph_type == "Gráfico de barras":
        col_x = st.selectbox("Variable X (Categórica/Agrupación):", all_cols, key="col_x_bar")
    else:
        col_x = None

with col3:
    if graph_type in ["Gráfico de dispersión (Scatterplot)", "Gráfico de línea"]:
        col_y = st.selectbox("Variable Y (Numérica/Continua):", numeric_cols, key="col_y_cont")
    elif graph_type == "Gráfico de barras":
        col_y = st.selectbox("Métrica Y (Recuento o Numérica):", all_cols, key="col_y_bar")
    else:
        col_y = None


# GENERADOR DE GRÁFICOS
if col_x is not None:
    st.write("### Visualización Generada")
    try:
        fig, ax = plt.subplots()
        # Configuración de estilo global para el plot (fuentes blancas)
        ax.set_facecolor(COLOR_BG)
        fig.patch.set_facecolor(COLOR_BG)
        plt.rcParams['text.color'] = 'white'
        plt.rcParams['axes.labelcolor'] = 'white'
        plt.rcParams['xtick.color'] = 'white'
        plt.rcParams['ytick.color'] = 'white'

        if graph_type == "Histograma":
            if df_nba[col_x].dtype in ['int64', 'float64']:
                sns.histplot(df_nba[col_x].dropna(), kde=True, color=COLOR_ACCENT, ax=ax)
                ax.set_title(f"Distribución de {col_x}", color=COLOR_3)
            else:
                 st.warning(f"'{col_x}' no es una variable numérica para Histograma.")

        elif graph_type == "Gráfico de dispersión (Scatterplot)" and col_y is not None:
            if df_nba[col_x].dtype in ['int64', 'float64'] and df_nba[col_y].dtype in ['int64', 'float64']:
                sns.scatterplot(data=df_nba.dropna(subset=[col_x, col_y]), x=col_x, y=col_y, color=COLOR_1, s=50, ax=ax)
                ax.set_title(f"Dispersión: {col_y} vs {col_x}", color=COLOR_3)
            else:
                 st.warning("Ambas variables para Scatterplot deben ser numéricas.")

        elif graph_type == "Gráfico de barras" and col_y is not None:
            if df_nba[col_y].dtype in ['int64', 'float64']:
                grouped_data = df_nba.groupby(col_x)[col_y].mean().reset_index()
                y_label = f"Media de {col_y}"
            else:
                grouped_data = df_nba[col_x].value_counts().reset_index()
                grouped_data.columns = [col_x, 'Conteo']
                col_y = 'Conteo'
                y_label = 'Conteo'

            y_col_name = col_y if 'Conteo' in grouped_data.columns else grouped_data.columns[-1]

            sns.barplot(data=grouped_data, x=col_x, y=y_col_name, palette="crest", ax=ax)
            ax.set_title(f"Gráfico de Barras por {col_x}", color=COLOR_3)
            ax.set_ylabel(y_label)
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()

        elif graph_type == "Gráfico de línea" and col_y is not None:
            if df_nba[col_x].dtype in ['int64', 'float64'] and df_nba[col_y].dtype in ['int64', 'float64']:
                sns.lineplot(data=df_nba.dropna(subset=[col_x, col_y]), x=col_x, y=col_y, color=COLOR_2, ax=ax)
                ax.set_title(f"Gráfico de Línea: {col_y} vs {col_x}", color=COLOR_3)
            else:
                 st.warning("Ambas variables para Gráfico de Línea deben ser numéricas.")

        # Ajuste final de estilos para el borde
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error al generar el gráfico. Verifica la selección de columnas: {e}")

# ============================
# 2. CLUSTERING KMEANS
# ============================

st.header("🎯 Clustering de Jugadores (Rendimiento)")

try:
    if not all(col in df_nba.columns for col in FEATURES_CLUSTER):
        st.warning(f"Omitiendo Clustering: Faltan columnas necesarias ({', '.join([col for col in FEATURES_CLUSTER if col not in df_nba.columns])}).")
    else:
        df_cluster_data = df_nba.dropna(subset=FEATURES_CLUSTER).copy()
        
        if df_cluster_data.empty:
            st.warning("No hay datos suficientes para realizar Clustering después de limpiar NaNs.")
        else:
            X_cluster = df_cluster_data[FEATURES_CLUSTER]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_cluster)

            kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
            df_cluster_data["cluster"] = kmeans.fit_predict(X_scaled)

            df_nba = df_nba.merge(
                df_cluster_data[['cluster']], 
                left_index=True, 
                right_index=True, 
                how='left'
            )
            
            st.write("### Centros de cada cluster (Valores Estandarizados)")
            st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=FEATURES_CLUSTER))

            summary = df_cluster_data.groupby("cluster")[FEATURES_CLUSTER].mean()
            dominant_feature = summary.idxmax(axis=1)

            interpretation_map = {
                "ts_pct_score": "Bajo uso, alta eficiencia",
                "dreb_pct_score": "Alto rebote defensivo",
                "ast_pct_score": "Creador de juego (AST)",
                "usg_pct_score": "Anotador de alto volumen",
                "net_rating_score": "Jugador de impacto neto positivo",
                "oreb_pct_score": "Alto rebote ofensivo"
            }

            df_nba["cluster_label"] = df_nba["cluster"].map(dominant_feature.map(interpretation_map)).fillna("N/A")

            st.write("### Jugadores de Ejemplo por Cluster")
            st.dataframe(df_nba[["player_name", "cluster", "cluster_label"]].sort_values('cluster').head(20))
    
except Exception as e:
    st.error(f"Error en el proceso de Clustering: {e}")

# ============================
# 3. REGRESIÓN / PREDICCIÓN DE RENDIMIENTO
# ============================

st.header("📈 Modelo Predictivo – Regresión del Rendimiento")

if model is None:
    st.warning("El modelo de Regresión no se pudo entrenar. Verifica las columnas de tu dataset.")
else:
    try:
        y_data = df_nba.dropna(subset=[TARGET] + FEATURES_REG)
        y = y_data[TARGET]
        X_data = y_data[FEATURES_REG]
        y_pred = model.predict(X_data) 

        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        col_metrica1, col_metrica2 = st.columns(2)
        with col_metrica1:
            st.metric(label="Error Absoluto Medio (MAE)", value=f"{mae:.4f}")
        with col_metrica2:
            st.metric(label="Coeficiente de Determinación (R²)", value=f"{r2:.4f}")

        # Coeficientes del modelo
        coef_df = pd.DataFrame({
            "Variable": FEATURES_REG,
            "Importancia": model.coef_
        }).sort_values("Importancia", ascending=False)

        st.write("### Importancia de variables (Coeficientes)")
        st.dataframe(coef_df)
        
        # Gráfico de Importancia
        fig, ax = plt.subplots(figsize=(8,5))
        ax.set_facecolor(COLOR_BG)
        fig.patch.set_facecolor(COLOR_BG)

        sns.barplot(data=coef_df, x="Importancia", y="Variable", palette="coolwarm", ax=ax)
        ax.set_title("Importancia de Variables en la Predicción", color=COLOR_3)
        ax.set_xlabel("Coeficiente (Impacto en Global Score)")
        ax.set_ylabel("Variable")
        
        # Ajustar color de ticks y bordes
        ax.tick_params(colors='white')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        st.pyplot(fig)


    except Exception as e:
        st.error(f"Error al evaluar el modelo de Regresión: {e}")


# ============================
# 4. TOP 10 JUGADORES
# ============================

st.header("🏆 Top 10 jugadores por rendimiento global")

try:
    if "player_name" in df_nba.columns and "team_abbreviation" in df_nba.columns and TARGET in df_nba.columns:
        top10 = (
            df_nba.sort_values(TARGET, ascending=False)
                .head(10)[["player_name","team_abbreviation",TARGET]]
        )

        st.table(top10)

        fig, ax = plt.subplots(figsize=(8,5))
        ax.set_facecolor(COLOR_BG)
        fig.patch.set_facecolor(COLOR_BG)
        
        sns.barplot(data=top10, y="player_name", x=TARGET, palette="crest", ax=ax)
        plt.title("Top 10 jugadores (Global Score)", color=COLOR_3)
        ax.set_xlabel("Global Score")
        ax.set_ylabel("Jugador")
        
        ax.tick_params(colors='white')
        ax.spines['left'].set_color('white')
        ax.spines['bottom'].set_color('white')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        st.pyplot(fig)
    else:
        st.warning(f"No se pudo generar el Top 10: Faltan columnas clave ('player_name', 'team_abbreviation' o '{TARGET}').")
    
except Exception as e:
    st.error(f"Error al generar el Top 10: {e}")

# ============================
# 5. DESCARGA DE RESULTADOS
# ============================

st.header("💾 Descarga")

if "cluster_label" in df_nba.columns:
    df_download = df_nba.drop(columns=['cluster_label'], errors='ignore')
else:
    df_download = df_nba
    
st.download_button(
    "📥 Descargar Dataset Modificado (con Clusters)",
    df_download.to_csv(index=False).encode('utf-8'),
    "nba_processed_with_clusters.csv",
    "text/csv"
)
