import pandas as pd
import numpy as np

def get_low_completion_columns(df: pd.DataFrame, seller_id_col: str = 'seller_id', threshold: int = 80) -> list:
    """
    Retorna las columnas que tienen un porcentaje de completitud menor al threshold especificado,
    calculado en relación a la cantidad de seller_id únicos.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame a analizar
    seller_id_col : str
        Nombre de la columna que contiene el seller_id
    threshold : int
        Porcentaje mínimo de completitud requerido (0-100)
    
    Returns:
    --------
    list
        Lista de columnas que no cumplen con el porcentaje mínimo de completitud
    """
    
    # Cantidad de sellers únicos
    total_sellers = df[seller_id_col].nunique()
    
    # Calcular porcentaje de completitud para cada columna
    completion_rates = (df.count() / total_sellers * 100).round(2)
    
    # Filtrar columnas por debajo del threshold
    low_completion_cols = completion_rates[completion_rates < threshold].index.tolist()
    
    return low_completion_cols

def preprocess_data(df_mkt_closed: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica los pasos de preprocesamiento al DataFrame combinado.
    
    Args:
        df_mkt_closed: DataFrame combinado de marketing leads y closed deals.

    Returns:
        DataFrame preprocesado.
    """
    # Usar el análisis de completitud
    low_quality_columns = get_low_completion_columns(df_mkt_closed, threshold=60)
    
    # Definir columnas a eliminar (incluyendo las identificadas por baja completitud)
    # Nota: Asegúrate de que 'sr_id' y 'won_date' no se eliminen si se usan en calculate_funnel_metrics
    columns_to_drop = ['landing_page_id', 'seller_id', 'sdr_id'] + low_quality_columns
    # Asegurarse de no eliminar columnas esenciales si están en la lista
    essential_cols = ['mql_id', 'sr_id', 'won_date', 'origin', 'first_contact_date'] 
    columns_to_drop = [col for col in columns_to_drop if col not in essential_cols and col in df_mkt_closed.columns]

    df_processed = df_mkt_closed.drop(columns=columns_to_drop)
    
    # Convertir fechas y crear nuevas columnas
    df_processed = df_processed.assign(
        first_contact_date=lambda df: pd.to_datetime(df['first_contact_date'], format='%Y-%m-%d', errors='coerce'),
        # Extraer solo la fecha de 'won_date' antes de convertir
        won_date_cleaned=lambda df: df['won_date'].astype(str).str[:10],
        won_date=lambda df: pd.to_datetime(df['won_date_cleaned'], format='%Y-%m-%d', errors='coerce'),
        target=lambda df: np.where(df['won_date'].isnull(), 0, 1),
        origin=lambda df: np.where(df['origin'].isnull(), 'unknown', df['origin'])
    ).drop(columns=['won_date_cleaned']) # Eliminar la columna auxiliar

    # Calcular days_to_convert solo si ambas fechas están presentes
    mask = df_processed['won_date'].notna() & df_processed['first_contact_date'].notna()
    df_processed.loc[mask, 'days_to_convert'] = (df_processed.loc[mask, 'won_date'] - df_processed.loc[mask, 'first_contact_date']).dt.days
    df_processed['days_to_convert'].fillna(np.nan, inplace=True) # Asegurar que los no calculados sean NaN

    return df_processed 