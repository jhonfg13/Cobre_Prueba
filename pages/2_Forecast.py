import streamlit as st
import pandas as pd
import os
from PIL import Image # To display images
import plotly.graph_objects as go

# --- Darts specific imports ---
try:
    from darts import TimeSeries
    from darts.models import ARIMA
    # Add other necessary imports from your notebook
except ImportError:
    st.error("Darts library not found. Please install it (`pip install darts`).")
    st.stop() # Stop execution if Darts is missing

# --- Import plotting functions ---
# Assuming plotting.py is in the root directory (one level up from pages)
# Adjust the path if plotting.py is elsewhere
import sys
# Add the root directory to the Python path to find plotting
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))) 
try:
    from plotting import plot_actual_vs_predicted_weekly, plot_aggregated_mqls_by_period
except ImportError:
    st.error("Could not import plotting functions from plotting.py. Make sure it exists in the project root.")
    # Define dummy functions to prevent NameErrors later if import fails
    def plot_actual_vs_predicted_weekly(*args, **kwargs): return go.Figure()
    def plot_aggregated_mqls_by_period(*args, **kwargs): return go.Figure()
    # Optionally add st.stop() here if plots are essential

# --- Constants ---
# Paths should be relative to the root directory where streamlit run app.py is executed
DATA_PATH_MKT = "data/raw/olist_marketing_qualified_leads_dataset.csv"
DATA_PATH_CLOSED = "data/raw/olist_closed_deals_dataset.csv"
STATIC_IMAGE_DIR = "outputs/figures"
FORECAST_STEPS = 12 # Number of steps to predict (e.g., 12 weeks)

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
        mql_weekly_series_pd = mql_daily_series_pd.resample('W-MON').sum() # O 'W-MON' para inicio de semana, etc.

        # Luego crear el TimeSeries de Darts desde mql_weekly_series_pd
        series_weekly = TimeSeries.from_dataframe(mql_weekly_series_pd.reset_index(), 
                                          time_col='first_contact_date', # O el nombre que quede tras reset_index
                                          value_cols='mql_count',
                                          freq='W-MON') # Asegúrate que la freq coincida
        # --- 3. Train Model (Use parameters from notebook) ---
        # >>> IMPORTANT: ADJUST THESE PARAMETERS TO MATCH YOUR FINAL MODEL <<<
        model = ARIMA(p=1, d=1, q=0, seasonal_order=(1, 1, 1, 4))
        model.fit(series_weekly)

        # --- 4. Predict ---
        forecast_ts = model.predict(n=forecast_n)

        # --- 5. Post-processing for Plots ---
        actuals_df = series_weekly.to_dataframe()
        forecast_df = forecast_ts.to_dataframe().round(0)
        #
        original_col_name = actuals_df.columns[0]
        actuals_df = actuals_df.rename(columns={original_col_name: 'Actual MQLs'})
        actuals_df['Data Type'] = 'Actual'
        # El forecast suele mantener el nombre de la columna original
        forecast_df = forecast_df.rename(columns={original_col_name: 'Predicted MQLs'})
        forecast_df['Data Type'] = 'Predicted'

        forecast_df_renamed = forecast_df.rename(columns={'Predicted MQLs': 'Actual MQLs'}) # Renombra para que coincida
        combined_df_rows = pd.concat([actuals_df, forecast_df_renamed], axis=0) 
        combined_df_rows = combined_df_rows[~combined_df_rows.index.duplicated(keep='first')]
        #
        combined_df_rows['contact_period'] = combined_df_rows.index.astype(str).str[:7]
        #
        df_predict_agg = combined_df_rows.groupby(['contact_period', 'Data Type'])\
            .agg(mql_count=('Actual MQLs', 'sum')).reset_index()

        return actuals_df, forecast_df, df_predict_agg

    except FileNotFoundError as e:
        st.error(f"Error loading data file: {e}. Please ensure data exists at the specified paths relative to the project root.")
        return None, None, None
    except ImportError: # Catch potential Darts import errors within the function too
        st.error("Darts library error during processing.")
        return None, None, None
    except Exception as e:
        st.error(f"An error occurred during data processing or modeling: {e}")
        return None, None, None

# --- Streamlit Page Layout ---

st.set_page_config(layout="wide") # Config can be set in app.py or here
st.title("Time Series Forecasting: MQLs")
st.markdown("Analysis and forecast of Marketing Qualified Leads (MQLs).")


st.header("Model Evaluation Plots")
st.write(f"Static images from model evaluation (source: `{STATIC_IMAGE_DIR}`):")

try:
    # Ensure path is correct relative to the root where streamlit is run
    abs_image_dir = os.path.abspath(STATIC_IMAGE_DIR)
    if not os.path.isdir(abs_image_dir):
         st.warning(f"Directory not found: `{abs_image_dir}`. Please check the `STATIC_IMAGE_DIR` path.")
         image_files = []
    else:
        image_files = [f for f in os.listdir(abs_image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    if not image_files:
        st.warning(f"No compatible image files found in `{abs_image_dir}`.")
    else:
        cols = st.columns(len(image_files) if len(image_files) <= 3 else 3)
        for i, img_file in enumerate(image_files):
            img_path = os.path.join(abs_image_dir, img_file)
            try:
                image = Image.open(img_path)
                cols[i % len(cols)].image(image, caption=img_file, use_column_width=True)
            except Exception as e:
                cols[i % len(cols)].error(f"Could not load image {img_file}: {e}")
except Exception as e:
    st.error(f"An error occurred accessing the image directory `{STATIC_IMAGE_DIR}`: {e}")


st.divider()
st.header("SARIMA Forecast Results")

# --- Generate Data and Plots ---
# Call the cached function
actual_data_df, forecast_data_df, aggregated_data_df = generate_forecast_data(
    DATA_PATH_MKT, DATA_PATH_CLOSED, FORECAST_STEPS
)

# Display Plot 1
st.subheader("Weekly Actuals vs. Forecast")
if actual_data_df is not None and forecast_data_df is not None:
    fig1 = plot_actual_vs_predicted_weekly(actual_data_df, forecast_data_df)
    st.plotly_chart(fig1, use_container_width=True)
else:
    st.warning("Could not generate data for the weekly actual vs predicted plot. Check data paths and processing steps.")

# Display Plot 2
st.subheader("Aggregated MQLs by Period (Actual vs. Forecast)")
if aggregated_data_df is not None:
    fig2 = plot_aggregated_mqls_by_period(aggregated_data_df)
    st.plotly_chart(fig2, use_container_width=True)
else:
     st.warning("Could not generate data for the aggregated MQL plot. Check data processing and aggregation steps.")


st.write("---")
st.markdown("Note: Forecasts generated using a SARIMA model based on historical weekly data. Ensure the model parameters in the code reflect the best model found during analysis.")
# You could potentially display the parameters used:
# st.caption(f"Model Parameters: p=1, d=1, q=0, seasonal_order=(1, 1, 1, 5)") # Example - make dynamic if possible