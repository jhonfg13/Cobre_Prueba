import pandas as pd
import os

def load_data(data_path: str = '../data/raw') -> pd.DataFrame:
    """
    Carga los datasets de leads y deals, los combina y devuelve un único DataFrame.

    Args:
        data_path (str): Ruta al directorio que contiene los archivos CSV.
                        Se espera encontrar 'olist_marketing_qualified_leads_dataset.csv' 
                        y 'olist_closed_deals_dataset.csv' en este directorio.

    Returns:
        pd.DataFrame: DataFrame combinado con la información de MQLs y Closed Deals.
                     Devuelve un DataFrame vacío si los archivos no se encuentran.
    """
    mkt_leads_path = os.path.join(data_path, 'olist_marketing_qualified_leads_dataset.csv')
    closed_deals_path = os.path.join(data_path, 'olist_closed_deals_dataset.csv')

    if not os.path.exists(mkt_leads_path) or not os.path.exists(closed_deals_path):
        print(f"Error: No se encontraron los archivos de datos en {data_path}")
        # Podríamos lanzar una excepción aquí también
        # raise FileNotFoundError(f"Archivos de datos no encontrados en {data_path}")
        return pd.DataFrame() # Devolver DataFrame vacío en caso de error

    try:
        df_mkt = pd.read_csv(mkt_leads_path)
        df_closed = pd.read_csv(closed_deals_path)
        
        # Combinar los dataframes
        df_mkt_closed = df_mkt.merge(df_closed, on='mql_id', how='left')
        
        return df_mkt_closed

    except Exception as e:
        print(f"Error al cargar o combinar los datos: {e}")
        return pd.DataFrame() # Devolver DataFrame vacío en caso de error 