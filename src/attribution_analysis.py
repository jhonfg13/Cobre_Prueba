import pandas as pd
import numpy as np

def calculate_origin_conversion(df_processed: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula las métricas de conversión agrupadas por el origen del lead.

    Args:
        df_processed: DataFrame preprocesado con información de leads y conversiones.

    Returns:
        DataFrame con métricas de conversión por origen.
    """
    origin_conversion = df_processed.query("origin != 'unknown'")\
                            .groupby('origin', as_index=False)\
                            .agg(
                                mql=('mql_id', 'count'),
                                won=('target', 'sum'),
                                days_to_convert_q3=('days_to_convert', lambda x: x.quantile(0.75)))
    # Calcular los porcentajes
    total_mql = origin_conversion['mql'].sum()
    total_won = origin_conversion['won'].sum()
    
    # Añadir columnas de porcentaje
    origin_conversion['mql_percentage'] = (origin_conversion['mql'] / total_mql * 100).round(2)
    origin_conversion['won_percentage'] = (origin_conversion['won'] / total_won * 100).round(2)
    origin_conversion['conversion'] = (origin_conversion['won'] / origin_conversion['mql']  * 100).round(2)
    origin_conversion['weighted_conversion'] = (origin_conversion['won'] / origin_conversion['mql']) * np.log(origin_conversion['mql'])
    
    origin_conversion = origin_conversion.sort_values(by='conversion', ascending=False)
    
    return origin_conversion

# Podríamos añadir aquí la función para calcular las métricas del funnel también.
def calculate_funnel_metrics(df_mkt_closed: pd.DataFrame) -> dict:
    """
    Calcula las métricas básicas del funnel de conversión.

    Args:
        df_mkt_closed: DataFrame con datos de MQL y Closed Deals.

    Returns:
        Un diccionario con las métricas del funnel (n_mql, n_sql, n_won, 
        conversion_mql_to_sql, conversion_sql_to_won, conversion_mql_to_won).
    """
    n_mql = df_mkt_closed['mql_id'].nunique()
    # Asumiendo que 'sr_id' indica un SQL (Sales Qualified Lead)
    n_sql = df_mkt_closed[df_mkt_closed['sr_id'].notna()]['mql_id'].nunique()
    # Asumiendo que 'won_date' indica una conversión (Won)
    n_won = df_mkt_closed[df_mkt_closed['won_date'].notna()]['mql_id'].nunique()

    # Evitar división por cero si no hay MQLs o SQLs
    conversion_mql_to_sql = (n_sql / n_mql) if n_mql > 0 else 0
    conversion_sql_to_won = (n_won / n_sql) if n_sql > 0 else 0
    conversion_mql_to_won = (n_won / n_mql) if n_mql > 0 else 0

    metrics = {
        "n_mql": n_mql,
        "n_sql": n_sql,
        "n_won": n_won,
        "conversion_mql_to_sql": conversion_mql_to_sql,
        "conversion_sql_to_won": conversion_sql_to_won,
        "conversion_mql_to_won": conversion_mql_to_won
    }
    
    return metrics

def calculate_origin_score(df: pd.DataFrame, weight_conversion: float = 0.6, weight_speed: float = 0.4) -> pd.DataFrame:
    """
    Calculates a combined score for origin performance based on
    weighted conversion efficiency and conversion speed (days_to_convert_q3).

    Args:
        df (pd.DataFrame): DataFrame containing 'origin', 'weighted_conversion',
                           and 'days_to_convert_q3'. Assumes it's the output
                           of calculate_origin_conversion or similar.
        weight_conversion (float): Weight for the weighted_conversion component (0 to 1).
        weight_speed (float): Weight for the days_to_convert_q3 component (0 to 1).
                                Should sum to 1 with weight_conversion.

    Returns:
        pd.DataFrame: Original DataFrame with added normalized columns and the final 'origin_score'.
                      Sorted by 'origin_score' descending. Returns only selected columns.
    """
    if not np.isclose(weight_conversion + weight_speed, 1.0):
        raise ValueError("Weights must sum to 1.0")

    # --- 1. Normalization (Manual Min-Max Scaling 0-1) ---
    # Create copies to avoid SettingWithCopyWarning if df is a slice
    df_scored = df.copy()

    # Handle cases where required columns might not exist or have NaNs
    if 'weighted_conversion' not in df_scored.columns or 'days_to_convert_q3' not in df_scored.columns:
        raise ValueError("Input DataFrame must contain 'weighted_conversion' and 'days_to_convert_q3' columns.")

    # Fill NaNs before normalization to avoid issues if a whole column is NaN
    # For days_to_convert_q3, filling with max might be more appropriate (worst case)
    df_scored['weighted_conversion'].fillna(0, inplace=True)
    # Check if days_to_convert_q3 has non-NaN values before filling with max
    if df_scored['days_to_convert_q3'].notna().any():
        max_days_val = df_scored['days_to_convert_q3'].max()
        df_scored['days_to_convert_q3'].fillna(max_days_val, inplace=True)
    else: # If all are NaN, fill with a default (e.g., 0 or handle differently)
        df_scored['days_to_convert_q3'].fillna(0, inplace=True) 


    # a) Weighted Conversion (Higher is better)
    min_wc = df_scored['weighted_conversion'].min()
    max_wc = df_scored['weighted_conversion'].max()
    if max_wc > min_wc: # Avoid division by zero
        df_scored['norm_weighted_conversion'] = (df_scored['weighted_conversion'] - min_wc) / (max_wc - min_wc)
    else:
        df_scored['norm_weighted_conversion'] = 0.5 # Assign mid-value if all are the same

    # b) Days to Convert Q3 (Lower is better - INVERTED scale)
    min_days = df_scored['days_to_convert_q3'].min()
    max_days = df_scored['days_to_convert_q3'].max()
    if max_days > min_days: # Avoid division by zero
        # Inverted scale: (max - value) / (max - min)
        df_scored['norm_days_to_convert'] = (max_days - df_scored['days_to_convert_q3']) / (max_days - min_days)
    else:
        df_scored['norm_days_to_convert'] = 0.5 # Assign mid-value if all are the same

    # Handle potential NaNs introduced during calculation (though previous fillna should prevent this)
    df_scored['norm_weighted_conversion'].fillna(0, inplace=True)
    df_scored['norm_days_to_convert'].fillna(0, inplace=True)


    # --- 2. Weighted Combination ---
    df_scored['origin_score'] = (weight_conversion * df_scored['norm_weighted_conversion'] + 
                                 weight_speed * df_scored['norm_days_to_convert'])
    #
    # Ensure all columns exist before selecting
    list_cols_sel = ['origin', 'mql', 'conversion', 'weighted_conversion', 'days_to_convert_q3', 'norm_weighted_conversion', 'norm_days_to_convert', 'origin_score']

    # --- 3. Return sorted DataFrame ---
    return df_scored.sort_values(by='origin_score', ascending=False)[list_cols_sel]