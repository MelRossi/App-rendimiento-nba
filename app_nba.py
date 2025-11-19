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
    /* Estilo para mover el logo arriba a la derecha */
    .logo-container {{
        display: flex;
        justify-content: flex-end;
        position: absolute;
        top: 20px;
        right: 20px;
        z-index: 999;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Contenedor para el logo en la esquina superior derecha
st.markdown('<div class="logo-container">', unsafe_allow_html=True)
# ASUME que "image_logo.png" está en la raíz de tu repositorio.
try:
    st.image("image_logo.png", width=120) 
except:
    st.warning("⚠️ Logo 'image_logo.png' no encontrado en el repositorio.")
st.markdown('</div>', unsafe_allow_html=True)

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
# CARGA DE DATOS (PREDETERMINADA)
# ============================

# Función para cargar datos con cache para eficiencia
@st.cache_data
def cargar_datos_predeterminados():
    try:
        # Asegúrate de que estos archivos estén en la raíz de tu repositorio
        data_files = {
            "all_seasons": pd.read_csv("all_seasons_filtrado.csv"),
            "player": pd.read_csv("player_filtrado.csv"),
            "team": pd.read_csv("team_filtrado.csv"),
            "line_score": pd.read_csv("line_score_filtrado.csv"),
            "game": pd.read_csv("game_filtrado.csv"),
            "df_nba": pd.read_csv("nba_puntaje_vara.csv"), # Este es el dataset principal para EDA/Modelos
        }
        st.success("✔️ Datos predeterminados cargados correctamente.")
        return data_files
    except FileNotFoundError as e:
        st.error(f"❌ Error al cargar archivo: {e}. Asegúrate de que el archivo CSV esté en la raíz del repositorio.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Ocurrió un error al cargar los datos: {e}")
        st.stop()


data_dict = cargar_datos_predeterminados()
# Asignamos el dataframe principal
df_nba = data_dict["df_nba"]


# ============================
# INICIO DE LA APLICACIÓN
# ============================

st.write("---") # Separador visual

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
    col_x = None
    col_y = None
    if graph_type == "Histograma":
        col_x = st.selectbox("Variable numerica:", numeric_cols)
    elif graph_type == "Boxplot":
        col_x = st.selectbox("Variable categórica:", categorical_cols)
        col_y = st.selectbox("Variable numérica:", numeric_cols)
    else:
        col_x = st.selectbox("Variable X:", df_nba.columns)
        col_y = st.selectbox("Variable Y:", df_nba.columns)

# GENERADOR DE GRÁFICOS
if col_x is not None:
    fig, ax = plt.subplots()
    
    if graph_type == "Histograma":
        sns.histplot(df_nba[col_x], kde=True, color=COLOR_ACCENT)
        plt.title(f"Histograma de {col_x}")
        
    elif graph_type == "Boxplot" and col_y is not None:
        sns.boxplot(data=df_nba, x=col_x, y=col_y, color=COLOR_1)
        plt.title(f"Boxplot de {col_y} por {col_x}")
        
    elif graph_type == "Scatterplot" and col_y is not None:
        sns.scatterplot(data=df_nba, x=col_x, y=col_y, color=COLOR_1)
        plt.title(f"Scatterplot de {col_x} vs {col_y}")

    elif graph_type == "Heatmap":
        # Aseguramos que sea solo numérico para correlación
        corr_data = df_nba.select_dtypes(include=['number']).corr()
        sns.heatmap(corr_data, annot=False, cmap="viridis")
        plt.title("Mapa de Calor de Correlación")
        
    st.pyplot(fig)


# ============================
# CLUSTERING KMEANS (Mantenemos esta sección)
# ============================

st.header("🎯 Clustering de Jugadores (Rendimiento)")

features = [
    'ts_pct_score','usg_pct_score','ast_pct_score',
    'oreb_pct_score','dreb_pct_score','net_rating_score'
]

X_cluster = df_nba[features] # Usamos X_cluster para no confundir con X de Regresión
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# KMeans ya fue entrenado en la versión original, lo mantendremos fijo en 4
kmeans = KMeans(n_clusters=4, random_state=42, n_init='auto')
df_nba["cluster"] = kmeans.fit_predict(X_scaled)

st.write("### Centros de cada cluster")
st.dataframe(pd.DataFrame(kmeans.cluster_centers_, columns=features))

# ... (El resto del código de Clustering se mantiene igual) ...
# INTERPRETACIÓN AUTOMÁTICA
st.write("### Interpretación automática de clusters:")

summary = df_nba.groupby("cluster")[features].mean()

dominant_feature = summary.idxmax(axis=1)

interpretation_map = {
    "ts_pct_score": "Bajo uso, alta eficiencia",
    "dreb_pct_score": "Alto rebote defensivo",
    "ast_pct_score": "Creador de juego (AST)",
    "usg_pct_score": "Anotador de alto volumen",
    "net_rating_score": "Impacto neto positivo" # Añadimos si es necesario
}

# Aquí usamos una función más robusta en caso de que el valor no esté mapeado
df_nba["cluster_label"] = dominant_feature.apply(lambda x: interpretation_map.get(x, f"Dominado por {x}"))

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

X_reg = df_nba[features_reg]
y = df_nba[target]

# Entrenar el modelo (para poder usar la importancia de variables y predecir)
model = LinearRegression()
model.fit(X_reg, y)

# Mostrar métricas del modelo entrenado
y_pred = model.predict(X_reg)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

st.write(f"**MAE del modelo general:** {mae:.4f}")
st.write(f"**R² del modelo general:** {r2:.4f}")

coef_df = pd.DataFrame({
    "Variable": features_reg,
    "Importancia": model.coef_
}).sort_values("Importancia", ascending=False)

st.write("### Importancia de variables")
st.dataframe(coef_df)

# ============================
# BARRA LATERAL: INPUTS PARA PREDICCIÓN INDIVIDUAL
# ============================

st.sidebar.header("🏀 Predecir Rendimiento Individual")
st.sidebar.markdown("---")

# Diccionario para almacenar los datos del usuario
datos_usuario = {}

# Valores medios para usar como default en el slider
media_valores = X_reg.mean()

for col in features_reg:
    # Usamos los min/max/mean del dataset para el rango del slider
    min_val = X_reg[col].min()
    max_val = X_reg[col].max()
    mean_val = media_valores[col]
    
    # Asume que todas las variables son numéricas (por el dataset original)
    if X_reg[col].dtype in ['float64', 'int64']:
        # Usamos 2 decimales para precisión en el paso
        step = 0.01 if max_val < 10 else 1 
        
        valor = st.sidebar.slider(
            f"Valor de: {col}", 
            float(min_val), 
            float(max_val), 
            float(mean_val), 
            step=step
        )
        datos_usuario[col] = valor
    else:
        st.sidebar.warning(f"La columna '{col}' no es numérica.")

# Crear un DataFrame con los datos del usuario (debe tener el mismo orden)
df_usuario = pd.DataFrame([datos_usuario])[features_reg]

# Realizar la predicción al presionar el botón
if st.sidebar.button("✨ Predecir Global Score"):
    try:
        prediccion = model.predict(df_usuario)
        
        st.sidebar.markdown("---")
        st.sidebar.success(
            f"**Puntaje Global (Predicción):**\n"
            f"{prediccion[0]:.4f}"
        )
        st.sidebar.write("*(El Global Score es un índice ponderado de eficiencia)*")

    except Exception as e:
        st.sidebar.error(f"Error en la predicción: {e}")


# ============================
# TOP 10 JUGADORES (Mantenemos esta sección)
# ============================

st.header("🏆 Top 10 jugadores por rendimiento global")
# ... (Se mantiene el código del Top 10) ...

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
# DESCARGA DE RESULTADOS (Mantenemos esta sección)
# ============================

st.download_button(
    "📥 Descargar Dataset Modificado",
    df_nba.to_csv(index=False),
    "nba_processed.csv",
    "text/csv"
)

