import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

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

def plot_actual_vs_predicted_weekly(actuals_df: pd.DataFrame, forecast_df: pd.DataFrame, title='Actual vs. Predicted Weekly MQLs'):
    """
    Generates a Plotly figure comparing actual weekly MQLs with predictions.
    Connects the last actual point to the first predicted point.

    Args:
        actuals_df (pd.DataFrame): DataFrame with actual data (needs DatetimeIndex and 'Actual MQLs' column).
        forecast_df (pd.DataFrame): DataFrame with predicted data (needs DatetimeIndex and 'Predicted MQLs' column).
                                      Should align chronologically after actuals_df.
        title (str): Title for the plot.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = go.Figure()

    actual_color = 'rgb(99, 110, 250)'
    predicted_color = 'rgb(239, 85, 59)'
    mql_col_actual = 'Actual MQLs'       # Assuming this is the column name
    mql_col_pred = 'Predicted MQLs'    # Assuming this is the column name

    # Ensure dataframes are not empty before plotting
    if not actuals_df.empty:
        fig.add_trace(go.Scatter(
            x=actuals_df.index,
            y=actuals_df[mql_col_actual],
            mode='lines',
            name='Actual',
            line=dict(color=actual_color)
        ))

    if not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df[mql_col_pred],
            mode='lines',
            name='Predicted',
            line=dict(color=predicted_color)
        ))

    # Add connecting line if both dataframes have data
    if not actuals_df.empty and not forecast_df.empty:
        last_actual_x = actuals_df.index[-1]
        last_actual_y = actuals_df[mql_col_actual].iloc[-1]
        first_predicted_x = forecast_df.index[0]
        first_predicted_y = forecast_df[mql_col_pred].iloc[0]

        fig.add_trace(go.Scatter(
            x=[last_actual_x, first_predicted_x],
            y=[last_actual_y, first_predicted_y],
            mode='lines',
            line=dict(color=predicted_color), # Match prediction line color
            showlegend=False
        ))

    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='MQL Count',
        legend_title='Data Type',
        hovermode="x unified"
    )
    return fig

def plot_aggregated_mqls_by_period(df_agg: pd.DataFrame, title='MQLs por contact_period'):
    """
    Generates a Plotly figure showing aggregated MQLs (Actual vs Predicted) by period.

    Args:
        df_agg (pd.DataFrame): DataFrame with columns like 'contact_period', 'mql_count', 'Data Type'.
        title (str): Title for the plot.

    Returns:
        go.Figure: Plotly figure object.
    """
    # Ensure the DataFrame and required columns exist
    if df_agg is None or df_agg.empty or not all(col in df_agg.columns for col in ['contact_period', 'mql_count', 'Data Type']):
        # Return an empty figure or a figure with a message if data is missing
        fig = go.Figure()
        fig.update_layout(title=title, xaxis={'visible': False}, yaxis={'visible': False},
                          annotations=[{'text': "Data for aggregation plot is missing.", 'xref': "paper", 'yref': "paper",
                                        'showarrow': False, 'font': {'size': 16}}])
        return fig
        
    fig = px.line(df_agg, x='contact_period', y='mql_count', color='Data Type', title=title,
                  labels={'mql_count': 'MQL Count', 'contact_period': 'Contact Period'})
    
    fig.update_layout(hovermode="x unified")
    
    return fig

def plot_actual_vs_predicted_weekly(actuals_df: pd.DataFrame, forecast_df: pd.DataFrame, title='Actual vs. Predicted Weekly MQLs'):
    """
    Generates a Plotly figure comparing actual weekly MQLs with predictions.
    Connects the last actual point to the first predicted point.

    Args:
        actuals_df (pd.DataFrame): DataFrame with actual data (needs DatetimeIndex and 'Actual MQLs' column).
        forecast_df (pd.DataFrame): DataFrame with predicted data (needs DatetimeIndex and 'Predicted MQLs' column).
                                      Should align chronologically after actuals_df.
        title (str): Title for the plot.

    Returns:
        go.Figure: Plotly figure object.
    """
    fig = go.Figure()

    actual_color = 'rgb(99, 110, 250)'
    predicted_color = 'rgb(239, 85, 59)'
    mql_col_actual = 'Actual MQLs'       # Assuming this is the column name
    mql_col_pred = 'Predicted MQLs'    # Assuming this is the column name

    # Ensure dataframes are not empty before plotting
    if actuals_df is not None and not actuals_df.empty and mql_col_actual in actuals_df.columns:
        fig.add_trace(go.Scatter(
            x=actuals_df.index,
            y=actuals_df[mql_col_actual],
            mode='lines',
            name='Actual',
            line=dict(color=actual_color)
        ))
    else:
         print("Warning: Actuals DataFrame is missing, empty, or lacks the required MQL column.")


    if forecast_df is not None and not forecast_df.empty and mql_col_pred in forecast_df.columns:
        fig.add_trace(go.Scatter(
            x=forecast_df.index,
            y=forecast_df[mql_col_pred],
            mode='lines',
            name='Predicted',
            line=dict(color=predicted_color)
        ))
    else:
        print("Warning: Forecast DataFrame is missing, empty, or lacks the required MQL column.")


    # Add connecting line if both dataframes have data and the required columns
    if actuals_df is not None and not actuals_df.empty and mql_col_actual in actuals_df.columns and \
       forecast_df is not None and not forecast_df.empty and mql_col_pred in forecast_df.columns:
        try:
            last_actual_x = actuals_df.index[-1]
            last_actual_y = actuals_df[mql_col_actual].iloc[-1]
            first_predicted_x = forecast_df.index[0]
            first_predicted_y = forecast_df[mql_col_pred].iloc[0]

            fig.add_trace(go.Scatter(
                x=[last_actual_x, first_predicted_x],
                y=[last_actual_y, first_predicted_y],
                mode='lines',
                line=dict(color=predicted_color), # Match prediction line color
                showlegend=False
            ))
        except IndexError:
             print("Warning: Could not create connecting line due to index issues.")


    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='MQL Count',
        legend_title='Data Type',
        hovermode="x unified"
    )
    return fig