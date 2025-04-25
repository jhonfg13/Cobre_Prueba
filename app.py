import streamlit as st
import pandas as pd
import os
from pathlib import Path
from PIL import Image # To display images
import plotly.graph_objects as go

# --- statsmodels specific imports ---
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    # Add other necessary imports from your notebook
except ImportError:
    st.error("statsmodels library not found. Please install it (`pip install statsmodels`).")
    st.stop() # Stop execution if statsmodels is missing

# --- Constants ---
# Paths should be relative to the root directory where streamlit run app.py is executed
DATA_PATH_MKT = "data/raw/olist_marketing_qualified_leads_dataset.csv"
DATA_PATH_CLOSED = "data/raw/olist_closed_deals_dataset.csv"
FORECAST_STEPS = 13

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
    from plotting import plot_days_to_convert_boxplot, plot_conversion_scatter, plot_actual_vs_predicted_weekly, plot_aggregated_mqls_by_period
except ImportError as e:
    st.error(f"Error al importar módulos. Asegúrate de que los archivos .py están en la carpeta 'src'. Detalle: {e}")
    st.stop() # Detener la ejecución si no se pueden importar los módulos

# --- Configuración de la página ---
st.set_page_config(
    page_title="Caso de Negocio: Análisis de Marketing",
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

# --- Cached Data Processing and Modeling Function ---
# (This function remains the same as previously defined)
@st.cache_data # Use st.cache_data for data/models
def generate_forecast_data(mkt_data_path, closed_data_path, forecast_n):
    """
    Loads data, preprocesses, trains SARIMA, predicts, and prepares DFs for plotting.
    """
    try:
        # --- 1. Load Data ---
        df_mkt = pd.read_csv(mkt_data_path)
        df_closed = pd.read_csv(closed_data_path)

        # --- 2. Preprocessing (Adapt from your notebook) ---
        df_mkt_closed = df_mkt.merge(df_closed, on='mql_id', how='left')
        df_processed = df_mkt_closed.assign(
            first_contact_date=lambda df: pd.to_datetime(df['first_contact_date'], errors='coerce'),
        ).dropna(subset=['first_contact_date'])
        # --- 3. Aggregate by day
        mql_daily_series = df_processed.groupby('first_contact_date', as_index=False).agg(mql_count=('mql_id', 'count'))\
                                .sort_values('first_contact_date')
        #
        # Reindexar a frecuencia diaria completa
        full_range = pd.date_range(start=mql_daily_series['first_contact_date'].min(), end=mql_daily_series['first_contact_date'].max(), freq='D')
        mql_daily_series = mql_daily_series.set_index('first_contact_date').reindex(full_range).fillna(0).rename_axis('first_contact_date').reset_index()
        # Asegúrate de que la columna de valores es numérica
        mql_daily_series['mql_count'] = mql_daily_series['mql_count'].astype(int)

        # Convertir a TimeSeries
        mql_daily_series_pd = mql_daily_series.query(" first_contact_date <= '2018-05-28' ").set_index('first_contact_date')
        mql_weekly_series_pd = mql_daily_series_pd.resample('W-MON').sum()

        # Suponiendo que `serie` es un pandas Series con índice de tiempo
        model = SARIMAX(mql_weekly_series_pd, order=(1, 1, 0), seasonal_order=(1, 1, 1, 4))
        # --- 3. Train Model (Use parameters from notebook) ---
        resultado = model.fit(disp=False)
        # --- 4. Predict ---
        forecast_ts = resultado.get_forecast(steps=forecast_n).predicted_mean
        # --- 5. Post-processing for Plots ---
        df_predicciones = forecast_ts.reset_index().round(0)
        df_predicciones.columns = ['first_contact_date', 'mql_count']
        df_predicciones['Data Type'] = 'Predicted'
        #
        mql_weekly_series_pd_reset = mql_weekly_series_pd.reset_index()
        mql_weekly_series_pd_reset['Data Type'] = 'Actual'
        #
        df_concat = pd.concat([mql_weekly_series_pd_reset, df_predicciones], axis=0)
        df_concat['contact_period'] = df_concat['first_contact_date'].astype(str).str[:7]
        #
        df_predict_agg = df_concat.groupby(['contact_period', 'Data Type'])\
                                    .agg(mql_count=('mql_count', 'sum')).reset_index()

        return mql_weekly_series_pd_reset, df_predicciones, df_predict_agg

    except FileNotFoundError as e:
        st.error(f"Error loading data file: {e}. Please ensure data exists at the specified paths relative to the project root.")
        return None, None, None
    except ImportError: # Catch potential Darts import errors within the function too
        st.error("Darts library error during processing.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred during data processing or modeling: {e}")
        return None, None, None

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

    st.markdown("Como se observa no tenemos las etapas de los diferentes canales de marketing, por lo que no podemos realizar un análisis de atribución completo.")

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
    st.markdown("""Se creo una **metodología adaptada al caso de negocio**, para poder realizar un análisis de atribución de marketing.
                    Se basa principalmente en capturar el tiempo de conversión de cada origen y el rendimiento de cada canal.""")

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
            st.caption("Gráfico de burbujas que segmenta el rendimiento del canal por peso de conversión y días para convertir, el tamaño de las burbujas esta dado por el score propuesto.")
        except Exception as e:
            st.warning(f"No se pudo generar el gráfico de rendimiento de canales: {e}")

    # --- Mostrar Datos Crudos (Opcional y colapsable) ---
    st.markdown("¿Quieres ver los datos de la **metodología** aplicada?")
    with st.expander("Ver Datos Preprocesados..."):
        st.dataframe(origin_score_df)

    # --- Sección: Forecast ---
    st.divider()
    st.header("Resultados de la Predicción de MQLs")
    st.markdown(
        """
        Bienvenid@ al **Forecast de MQLs**.

        En esta sección podrás ver los resultados de la predicción de MQLs:

        - 📈 **Evaluación de Frecuencia**: Se evaluó la frecuencia de los datos para poder realizar un pronóstico más preciso.
        - 📊 **Comparación de Modelos**: Se compararon los resultados de los diferentes modelos de pronóstico.
        - 📈 **Pronóstico**: Se realizó un pronóstico de los MQLs para los próximos 3 meses.

        Explora los resultados para tomar decisiones informadas basadas en datos.
    """
    )

    st.title("Forecasting")
    st.markdown("Dentro del análisis se decidio agrupar los datos por semana, para poder realizar un pronóstico más preciso")


    st.header("Gráficas de Evaluación de Modelos")

    #
    fig_dir = os.path.join(current_dir, 'outputs', 'figures')
    st.write(f"Static images from model evaluation (source: `{fig_dir}`):")
    names_fig = ['comparacion_modelos_pronostico.png', 'acf_mqls_semanales.png',
                 'comparacion_modelos_pronostico_semanales.png', 'sarimax_analisis_residuos_semanales.png']
    image_files = [os.path.join(fig_dir, name) for name in names_fig]
    # Crear contenedores para mostrar las imágenes
    cols = st.columns(min(4, len(image_files)))
    for i, img_file in enumerate(image_files):
        try:
            image = Image.open(img_file)
            cols[i % len(cols)].image(image, caption=os.path.basename(img_file), use_container_width=True)
        except Exception as e:
            cols[i % len(cols)].error(f"Could not load image {img_file}: {e}")

    # Separador y siguiente sección
    st.divider()
    st.header("SARIMA Resultados de la Predicción")

    # --- Generate Data and Plots ---
    # Call the cached function
    actual_data_df, forecast_data_df, aggregated_data_df = generate_forecast_data(
        DATA_PATH_MKT, DATA_PATH_CLOSED, FORECAST_STEPS)

    # Display Plot 1
    st.subheader("Real Semanal vs. Predicción")
    if actual_data_df is not None and forecast_data_df is not None:
        fig1 = plot_actual_vs_predicted_weekly(actual_data_df, forecast_data_df)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.warning("Could not generate data for the weekly actual vs predicted plot. Check data paths and processing steps.")

    # Display Plot 2
    st.subheader("MQLs Agregados por Periodo (Real vs. Predicción)")
    if aggregated_data_df is not None:
        fig2 = plot_aggregated_mqls_by_period(aggregated_data_df)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.warning("Could not generate data for the aggregated MQL plot. Check data processing and aggregation steps.")
        
    st.write("---")
    st.markdown("""Nota: Los pronósticos se generaron con un modelo SARIMAX sobre la serie semanal mql_weekly_series_pd, usando order=(1, 1, 0) y seasonal_order=(1, 1, 1, 4).
                El modelo captura tanto la tendencia como la estacionalidad cada 4 semanas, utilizando componentes autorregresivos y de diferencia para estabilizar la serie.""")
# You can add other high-level elements or introductory content here.
# The navigation to the "Forecast" page will be handled automatically by Streamlit
# because of the file in the pages/ directory.

else:
    st.warning(f"La aplicación no puede continuar porque los datos no se cargaron correctamente. Se intentó cargar desde: {data_path_used}")