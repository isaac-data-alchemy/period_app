import pandas as pd
import streamlit as st


def convert_to_datetime(df, column_name):
    """
    Convert a column in a DataFrame to datetime format.

    Args:
        df (pd.DataFrame): The DataFrame containing the column to be converted.
        column_name (str): The name of the column to convert to datetime.

    Raises:
        ValueError: If some values in the column cannot be converted to datetime.

    Displays:
        Error message if conversion fails.

    """
    try:
        df[column_name] = pd.to_datetime(
            df[column_name], format="%Y-%m-%d", errors="coerce"
        )
        if df[column_name].isnull().any():
            raise ValueError(
                f"Some values in '{column_name}' could not be converted to datetime."
            )
    except Exception as e:
        st.error(f"Error in converting '{column_name}': {e}0")
