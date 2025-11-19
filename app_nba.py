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
    .stSlider label, .stSelectbox label, .stDownloadButton, .stButton, .stMarkdown, .stTable, .dataframe-content, .stTextInput {{
        color: white !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================
# CARGA DE DATOS Y LÓGICA DE MODELADO
# ============================

# Variables de la Regresión (de tu código)
TARGET = "global_score"
FEATURES_REG = [
    "ts_pct_score","usg_pct_score","dreb_pct_score","ast_pct_score",
    "oreb_pct_score","age","player_height","player_weight"
]
# Columnas necesarias para Clustering
FEATURES_CLUSTER = [
    'ts_pct_score','usg_pct_score','ast_pct_score',
    'oreb_pct_score','dreb_pct_score','net_rating_score'
]

# Función para cargar datos con cache
@st.cache_data
def load_data(uploaded_file):
    """Carga el dataset principal para EDA y modelado."""
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error al cargar el archivo CSV: {e}")
            return pd.DataFrame() 
    return pd.DataFrame()

# Inicialización y entrenamiento del modelo de Regresión (se hace una sola vez)
@st.cache_resource
def train_regression_model(data, features, target_col):
    """Entrena y devuelve el modelo de Regresión Lineal y el conjunto de referencia."""
    try:
        # Verifica que todas las columnas necesarias existan
        required_cols = features + [target_col]
        if not all(col in data.columns for col in required_cols):
            st.error("Error: El dataset no contiene todas las columnas necesarias para el modelo de Regresión.")
            return None, None

        X = data[features]
        y = data[target_col]
        
        model = LinearRegression()
        model.fit(X, y)
        return model, X
    except Exception as e:
        st.error(f"Error al entrenar el modelo de Regresión: {e}")
        return None, None

# ============================
# BARRA LATERAL: CARGA Y PREDICCIÓN MANUAL
# ============================

st.sidebar.markdown("<p class='sidebar-header'>📂 Carga de Datos</p>", unsafe_allow_html=True)
uploaded_file = st.sidebar.file_uploader("Sube tu archivo 'nba_puntaje_vara.csv'", type=['csv'])

df_nba = load_data(uploaded_file)

# Detiene la ejecución si los datos no están cargados correctamente
if df_nba.empty or not all(col in df_nba.columns for col in FEATURES_REG + [TARGET]):
    st.sidebar.error("Esperando la carga del archivo CSV con columnas correctas.")
    st.stop()
else:
    st.sidebar.success("✔️ Dataset cargado correctamente.")

# Entrenar el modelo una vez que el DF está listo
model, X_ref = train_regression_model(df_nba, FEATURES_REG, TARGET)


st.sidebar.markdown("---")
st.sidebar.markdown("<p class='sidebar-header'>🚀 Predicción Individual</p>", unsafe_allow_html=True)

datos_usuario = {}

# Crear un diccionario de rangos y valores medios para los sliders
rangos = {}
for col in FEATURES_REG:
    if col in df_nba.columns:
        rangos[col] = {
            "min": df_nba[col].min(),
            "max": df_nba[col].max(),
            "mean": df_nba[col].mean()
        }
    else:
        # Esto no debería pasar si la verificación de columnas fue exitosa, pero es un fallback
        rangos[col] = {"min": 0, "max": 1, "mean": 0.5}

# Crear los Sliders en la barra lateral
for col in FEATURES_REG:
    rango = rangos[col]
    # Determinar el paso (step) del slider
    if 'score' in col or 'pct' in col:
        step_val = 0.01 
        fmt = "%.2f"
    else:
        step_val = 1.0 
        fmt = "%g"
    
    # Asegurar que el valor inicial esté dentro del rango
    mean_val_safe = max(rango["min"], min(rango["max"], rango["mean"]))
    
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
    # 1. Convertir el input del usuario a DataFrame
    df_usuario = pd.DataFrame([datos_usuario])
    
    # 2. Asegurar el orden de las columnas coincida
    df_usuario_final = df_usuario[FEATURES_REG]

    try:
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
        st.sidebar.error(f"Error en la predicción. Asegúrate que todas las variables son numéricas: {e}")

st.sidebar.markdown("---")

# ============================
# CONTENIDO PRINCIPAL: TÍTULO Y SECCIONES
# ============================

# Colocar aquí el logo 
 st.markdown('<div class="logo-container">', unsafe_allow_html=True)
 try:
     st.image("Image_logo.png", width=120) 
 except:
     pass
 st.markdown('</div>', unsafe_allow_html=True)


st.markdown("<h1 class='title'>🏀 NBA Performance Analytics</h1>", unsafe_allow_html=True)

st.markdown(
    f"""
    <h4 style="color:{COLOR_3}; text-align:center;">
        Análisis avanzado de rendimiento y clusters
    </h4>
    """,
    unsafe_allow_html=True,
)


# ============================
# 1. EDA INTERACTIVO (Gráficos Ajustados)
# ============================

st.header("📊 Exploración de Datos (EDA)")

col1, col2, col3 = st.columns([1, 1, 1])

numeric_cols = df_nba.select_dtypes(include=['number']).columns
all_cols = df_nba.columns

with col1:
    # Tipos de gráficos solicitados: Histograma, Scatterplot, Barras, Línea
    graph_type = st.selectbox(
        "Tipo de gráfico:",
        ["Histograma", "Gráfico de dispersión (Scatterplot)", "Gráfico de barras", "Gráfico de línea"]
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
        # Para Scatter/Línea, Y debe ser numérica
        col_y = st.selectbox("Variable Y (Numérica/Continua):", numeric_cols, key="col_y_cont")
    elif graph_type == "Gráfico de barras":
        # Para Barras, Y debe ser la métrica a contar/sumar
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
            sns.histplot(df_nba[col_x], kde=True, color=COLOR_ACCENT, ax=ax)
            ax.set_title(f"Distribución de {col_x}", color=COLOR_3)

        elif graph_type == "Gráfico de dispersión (Scatterplot)" and col_y is not None:
            sns.scatterplot(data=df_nba, x=col_x, y=col_y, color=COLOR_1, s=50, ax=ax)
            ax.set_title(f"Dispersión: {col_y} vs {col_x}", color=COLOR_3)

        elif graph_type == "Gráfico de barras" and col_y is not None:
            # Lógica para barras: usar el recuento si Y no es numérica, o la media si lo es
            if df_nba[col_y].dtype in ['int64', 'float64']:
                grouped_data = df_nba.groupby(col_x)[col_y].mean().reset_index()
                y_label = f"Media de {col_y}"
            else:
                grouped_data = df_nba[col_x].value_counts().reset_index()
                grouped_data.columns = [col_x, 'Conteo']
                col_y = 'Conteo'
                y_label = 'Conteo'

            sns.barplot(data=grouped_data, x=col_x, y=col_y, palette="crest", ax=ax)
            ax.set_title(f"Gráfico de Barras por {col_x}", color=COLOR_3)
            ax.set_ylabel(y_label)
            ax.tick_params(axis='x', rotation=45)
            plt.tight_layout()

        elif graph_type == "Gráfico de línea" and col_y is not None:
            sns.lineplot(data=df_nba, x=col_x, y=col_y, color=COLOR_2, ax=ax)
            ax.set_title(f"Gráfico de Línea: {col_y} vs {col_x}", color=COLOR_3)

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
    X_cluster = df_nba[FEATURES_CLUSTER]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)

    kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
    df_nba["cluster"] = kmeans.fit_predict(X_scaled)

    st.write("### Centros de cada cluster (Valores Estandarizados)")
    st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=FEATURES_CLUSTER))

    # Interpretación automática (se mantiene la lógica)
    summary = df_nba.groupby("cluster")[FEATURES_CLUSTER].mean()
    dominant_feature = summary.idxmax(axis=1)

    interpretation_map = {
        "ts_pct_score": "Bajo uso, alta eficiencia",
        "dreb_pct_score": "Alto rebote defensivo",
        "ast_pct_score": "Creador de juego (AST)",
        "usg_pct_score": "Anotador de alto volumen",
        "net_rating_score": "Jugador de impacto neto positivo",
        "oreb_pct_score": "Alto rebote ofensivo"
    }

    df_nba["cluster_label"] = dominant_feature.map(interpretation_map).fillna("Otros")

    st.write("### Jugadores de Ejemplo por Cluster")
    st.dataframe(df_nba[["player_name", "cluster", "cluster_label"]].sort_values('cluster').head(20))
    
except KeyError:
    st.error("Error: Faltan columnas necesarias para el Clustering. Asegúrate de que el CSV subido las contenga.")
except Exception as e:
    st.error(f"Error en el proceso de Clustering: {e}")

# ============================
# 3. REGRESIÓN / PREDICCIÓN DE RENDIMIENTO
# ============================

st.header("📈 Modelo Predictivo – Regresión del Rendimiento")

try:
    # Evaluación del modelo ya entrenado
    y = df_nba[TARGET]
    y_pred = model.predict(X_ref) 

    mae = mean_absolute_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    st.write(f"**Error Absoluto Medio (MAE):** {mae:.4f}")
    st.write(f"**Coeficiente de Determinación (R²):** {r2:.4f}")

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


except KeyError:
    st.error("Error: Faltan columnas necesarias para evaluar el modelo de Regresión.")
except Exception as e:
    st.error(f"Error al evaluar el modelo de Regresión: {e}")


# ============================
# 4. TOP 10 JUGADORES
# ============================

st.header("🏆 Top 10 jugadores por rendimiento global")

try:
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
    
    # Ajustar color de ticks y bordes
    ax.tick_params(colors='white')
    ax.spines['left'].set_color('white')
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    st.pyplot(fig)
    
except KeyError:
    st.warning(f"No se pudo generar el Top 10: La columna '{TARGET}' o 'player_name' no se encontró.")

# ============================
# 5. DESCARGA DE RESULTADOS
# ============================

st.header("💾 Descarga")

st.download_button(
    "📥 Descargar Dataset Modificado (con Clusters)",
    df_nba.to_csv(index=False).encode('utf-8'),
    "nba_processed_with_clusters.csv",
    "text/csv"
)

