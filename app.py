import streamlit as st
import pandas as pd
import os

# Ajustar la ruta para importar desde la carpeta 'src'
import sys
# Obtener la ruta absoluta del directorio actual del script (app.py)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Obtener la ruta absoluta del directorio 'src'
src_dir = os.path.join(current_dir, 'src')
# Añadir el directorio 'src' al sys.path
if src_dir not in sys.path:
    sys.path.append(src_dir)

# Ahora podemos importar los módulos
try:
    from data_loader import load_data
    from preprocessing import preprocess_data
    from attribution_analysis import calculate_origin_conversion, calculate_funnel_metrics, calculate_origin_score
    from plotting import plot_days_to_convert_boxplot, plot_conversion_scatter
except ImportError as e:
    st.error(f"Error al importar módulos. Asegúrate de que los archivos .py están en la carpeta 'src'. Detalle: {e}")
    st.stop() # Detener la ejecución si no se pueden importar los módulos

# --- Configuración de la página ---
st.set_page_config(
    page_title="Análisis de Atribución de Marketing",
    page_icon="📊",
    layout="wide" # Usar layout ancho para mejor visualización de tablas y gráficos
)

# --- Título de la Aplicación ---
st.title("📊 Reporte de Atribución y Conversión de Marketing")
st.markdown("Análisis del origen de los leads y su impacto en la conversión.")

# --- Carga y Procesamiento de Datos (con Caching) ---
@st.cache_data # Usar cache para evitar recargar/reprocesar datos en cada interacción
def load_and_process_data():
    # Determinar la ruta a la carpeta 'data' relativa a la ubicación de app.py
    # ASUMIENDO: app.py está en 'marketing_analytics_case' y 'data' está dentro de 'marketing_analytics_case'
    # current_dir = os.path.dirname(os.path.abspath(__file__)) # Ya definido globalmente
    data_dir = os.path.join(current_dir, 'data', 'raw') # <-- CORRECCIÓN AQUÍ
    df_raw = load_data(data_path=data_dir)
    if df_raw.empty:
        st.error(f"No se pudieron cargar los datos desde {data_dir}. Verifica la ruta y los archivos CSV.")
        return None, None, None # Devolver None si hay error
    df_processed = preprocess_data(df_raw.copy()) # Usar una copia para evitar modificar el df original en caché
    return df_raw, df_processed, data_dir # Devolvemos data_dir para debugging si es necesario

df_raw, df_processed, data_path_used = load_and_process_data()

# --- Ejecución del Análisis y Visualización (si los datos se cargaron) ---
if df_processed is not None:

    # --- Sección: Resumen del Funnel ---
    st.header("📉 Resumen del Funnel de Conversión")
    # Pasamos el df_mkt_closed original a calculate_funnel_metrics
    # Asegúrate de que df_raw tiene las columnas necesarias ('mql_id', 'sr_id', 'won_date')
    # O ajusta calculate_funnel_metrics para usar df_processed si es más conveniente
    # Revisando el código, df_processed ya contiene las columnas necesarias después del merge y antes del drop.
    funnel_metrics = calculate_funnel_metrics(df_raw) # Usamos df_raw que es el merge inicial

    col1, col2, col3 = st.columns(3)
    col1.metric("Leads Calificados (MQL)", f"{funnel_metrics['n_mql']:,}")
    col2.metric("Oportunidades (SQL)", f"{funnel_metrics['n_sql']:,}")
    col3.metric("Clientes Ganados (Won)", f"{funnel_metrics['n_won']:,}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Tasa MQL -> SQL", f"{funnel_metrics['conversion_mql_to_sql']:.2%}")
    col5.metric("Tasa SQL -> Won", f"{funnel_metrics['conversion_sql_to_won']:.2%}")
    col6.metric("Tasa MQL -> Won (Global)", f"{funnel_metrics['conversion_mql_to_won']:.2%}")

    st.markdown("---") # Separador

    # --- Sección: Análisis por Origen ---
    st.header("🎯 Análisis de Conversión por Origen")
    origin_conversion_df = calculate_origin_conversion(df_processed)
    origin_score_df = calculate_origin_score(origin_conversion_df)

    st.subheader("Tabla de Métricas por Origen")
    # Asegurarse de que las columnas existen antes de aplicar formato
    format_dict = {}
    if 'mql_percentage' in origin_conversion_df.columns: format_dict['mql_percentage'] = "{:.2f}%"
    if 'won_percentage' in origin_conversion_df.columns: format_dict['won_percentage'] = "{:.2f}%"
    if 'conversion' in origin_conversion_df.columns: format_dict['conversion'] = "{:.2f}%"
    if 'weighted_conversion' in origin_conversion_df.columns: format_dict['weighted_conversion'] = "{:.4f}"
    if 'days_to_convert_q3' in origin_conversion_df.columns: format_dict['days_to_convert_q3'] = "{:.1f}"
    
    st.dataframe(origin_conversion_df.style.format(format_dict))
    st.caption("La tabla muestra MQLs, Leads Ganados (Won), el percentil 75 de días para convertir, porcentajes de MQL y Won sobre el total, tasa de conversión y conversión ponderada por volumen.")

    st.markdown("---") # Separador

    # --- Sección: Visualizaciones ---
    st.header("📈 Visualizaciones Clave")

    col_a, col_b = st.columns(2)

    with col_a:
        st.subheader("Distribución del Tiempo de Conversión")
        try:
            fig_boxplot = plot_days_to_convert_boxplot(df_processed)
            st.plotly_chart(fig_boxplot, use_container_width=True)
            st.caption("Boxplot mostrando la mediana, cuartiles y outliers de los días necesarios para convertir un lead, según su origen.")
        except Exception as e:
            st.warning(f"No se pudo generar el gráfico de distribución de tiempo: {e}")

    with col_b:
        st.subheader("Rendimiento de Canales por Peso de Conversión y Días para Convertir")
        try:
            fig_scatter = plot_conversion_scatter(origin_score_df)
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Gráfico de dispersión que relaciona el número de MQLs con la tasa de conversión. El tamaño de las burbujas representa el número de leads ganados.")
        except Exception as e:
            st.warning(f"No se pudo generar el gráfico de rendimiento de canales: {e}")

    # --- Mostrar Datos Crudos (Opcional y colapsable) ---
    with st.expander("Ver Datos Preprocesados"):
        st.dataframe(origin_score_df)

    # --- Sección: Forecast ---
    st.set_page_config(
    page_title="Marketing Analytics",
    page_icon="📊", # Optional: Add an icon
    layout="wide")

    st.title("Marketing Analytics Dashboard")
    st.markdown(
        """
        Welcome to the Marketing Analytics Dashboard.

        Use the navigation sidebar on the left to explore different sections:
        - **Forecast:** View MQL time series analysis and forecasts.
    
        *(Add more pages/sections as needed)*
    """)

# You can add other high-level elements or introductory content here.
# The navigation to the "Forecast" page will be handled automatically by Streamlit
# because of the file in the pages/ directory.

else:
    st.warning(f"La aplicación no puede continuar porque los datos no se cargaron correctamente. Se intentó cargar desde: {data_path_used}")