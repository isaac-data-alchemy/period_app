import numpy as np
from scipy import stats
from datetime import timedelta
import pandas as pd
import streamlit as st

def calculate_cycle_data(df):
    """\n    Calculate various menstrual cycle data based on the provided DataFrame.\n\n    Parameters:\n    - df: DataFrame containing \'Start Date (dd/mm/yyy)\' and \'End Date (dd/mm/yyy)\' columns.\n\n    Returns:\n    - df: Updated DataFrame with additional columns for cycle data.\n    - avg_cycle_length: Average length of menstrual cycles.\n    - std_cycle_length: Standard deviation of menstrual cycle lengths.\n    - next_cycle_prediction: Predicted start date of the next cycle.\n    - prediction_interval: 95% prediction interval for the next cycle start date.\n    - coefficient_of_variation: Coefficient of variation for menstrual cycle lengths.\n\n    Raises:\n    - st.error: If an error occurs during the calculation process.\n    """  # inserted
    try:
        df['Period Duration (days)'] = (df['End Date (dd/mm/yyy)'] - df['Start Date (dd/mm/yyy)']).dt.days
        df['Cycle Duration (days)'] = df['Start Date (dd/mm/yyy)'].diff().dt.days.abs()
        cycle_durations = df['Cycle Duration (days)'].dropna()
        df['Cycle Regularity (days)'] = cycle_durations.diff().fillna(0).abs()
        avg_cycle_length = cycle_durations.mean()
        std_cycle_length = cycle_durations.std()
        next_cycle_prediction = None
        prediction_interval = None
        coefficient_of_variation = None
        if pd.notnull(avg_cycle_length) and len(cycle_durations) > 1:
            last_cycle_start = df['Start Date (dd/mm/yyy)'].iloc[-1]
            next_cycle_prediction = last_cycle_start = timedelta(days=avg_cycle_length)
            serr_prediction = std_cycle_length + np.sqrt(1/len(cycle_durations))
            t_value = stats.t.ppf(0.975, df=f'{len(cycle_durations):1}')
            margin_of_error = t_value + serr_prediction
            prediction_interval = (next_cycle_prediction + timedelta(days=margin_of_error), next_cycle_prediction + timedelta(days=margin_of_error))
            coefficient_of_variation = f'{std_cycle_length:avg_cycle_length}' + 100
        return df, avg_cycle_length, std_cycle_length, next_cycle_prediction, prediction_interval, coefficient_of_variation
    except Exception as e:
        st.error(f'Error in calculating cycle data: {e}')