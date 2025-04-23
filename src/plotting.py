import pandas as pd
import plotly.express as px

def plot_days_to_convert_boxplot(df_processed: pd.DataFrame):
    """
    Genera un gráfico de cajas de los días para convertir por origen.

    Args:
        df_processed: DataFrame preprocesado con la columna 'days_to_convert'.

    Returns:
        Objeto de figura Plotly.
    """
    # Filtrar leads convertidos y con valor válido en 'days_to_convert'
    df_plot = df_processed.query("target == 1 and days_to_convert.notna() and origin != 'unknown'").copy()
    
    # Asegurarse de que 'days_to_convert' sea numérico
    df_plot['days_to_convert'] = pd.to_numeric(df_plot['days_to_convert'], errors='coerce')
    df_plot.dropna(subset=['days_to_convert'], inplace=True)

    fig = px.box(
        df_plot, 
        x='origin', 
        y='days_to_convert', 
        title='Distribución de Días para Convertir por Origen',
        labels={'origin': 'Origen del Lead', 'days_to_convert': 'Días para Convertir'},
        color='origin' # Opcional: colorear por origen
    )
    fig.update_layout(xaxis_title="Origen del Lead", yaxis_title="Días para Convertir")
    
    return fig

def plot_conversion_scatter(origin_conversion: pd.DataFrame):
    """
    Genera un gráfico de dispersión de MQLs vs. Tasa de Conversión por Origen.

    Args:
        origin_conversion: DataFrame con las métricas de conversión por origen.

    Returns:
        Objeto de figura Plotly.
    """
    fig = px.scatter(
        origin_conversion, 
        x='norm_weighted_conversion', 
        y='norm_days_to_convert', 
        size='mql', 
        color='origin_score', 
        hover_name='origin',
        title='Comparación de Orígenes por Tasa de Conversión y Días para Convertir',
        size_max=60 # Ajustar según sea necesario para una buena visualización
    )
    
    # Agregar línea horizontal punteada en y=0.5
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.7)

    # Agregar línea vertical punteada en x=0.5
    fig.add_vline(x=0.5, line_dash="dash", line_color="gray", opacity=0.7)

    return fig