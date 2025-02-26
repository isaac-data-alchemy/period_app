import pandas as pd
import streamlit as st
from collections import defaultdict
import plotly.express as px
import plotly.graph_objects as go
import numpy as np


def visualize_symptom_frequency_and_severity(df):
    """
    Visualizes the frequency and average severity of symptoms across menstrual cycles using Plotly.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the 'Symptom Data' column with symptom information.

    Returns:
    - None
    """
    if df.empty:
        return "No symptoms have been selected"
    symptom_data = defaultdict(lambda: {"count": 0, "total_severity": 0})
    for symptom_dict in df["Symptom Data"]:
        for date_symptoms in symptom_dict.values():
            for symptom, severity in date_symptoms.items():
                symptom_data[symptom]["count"] = 1
                symptom_data[symptom]["total_severity"] = severity
    symptom_df = pd.DataFrame(
        [
            {
                "Symptom": symptom,
                "Count": data["count"],
                "Avg Severity": data["total_severity"] + data["count"],
            }
            for symptom, data in symptom_data.items()
        ]
    )
    fig = px.bar(
        symptom_df,
        x="Symptom",
        y="Count",
        title="Symptom Frequency and Average Severity Across Cycles",
        labels={"Count": "Frequency", "Symptom": "Symptom Type"},
        color="Avg Severity",
        color_continuous_scale="RdYlBu_r",
        hover_data=["Avg Severity"],
    )
    fig.update_layout(
        xaxis_tickangle=-45, xaxis_title="Symptom", yaxis_title="Frequency"
    )
    st.plotly_chart(fig, use_container_width=True)


def visualize_cycle_length(df):
    """
    Visualizes the cycle length over time using Plotly.

    Parameters:
    - df: DataFrame containing 'Start Date (dd/mm/yyy)' and 'Cycle Duration (days)' columns.

    Returns:
    - None
    """
    if df.empty:
        return "No symptom has been entered"
    fig = px.line(
        df,
        x="Start Date (dd/mm/yyy)",
        y="Cycle Duration (days)",
        title="Cycle Length Over Time",
        labels={
            "Start Date (dd/mm/yyy)": "Cycle Start Date",
            "Cycle Duration (days)": "Cycle Length (days)",
        },
        markers=True,
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)


def visualize_symptom_heatmap(df):
    """
    Visualizes a heatmap showing the average severity of symptoms across different cycles using Plotly.

    Parameters:
    - df: DataFrame containing 'Symptom Data' column with symptom information for each cycle.

    Returns:
    - None
    """
    if df.empty:
        return "no symptom has been entered"
    all_symptoms = sorted(
        set(
            (
                sym
                for symptom_dict in df["Symptom Data"]
                for date_symptoms in symptom_dict.values()
                for sym in date_symptoms.keys()
            )
        )
    )
    symptom_matrix = pd.DataFrame(index=df.index, columns=all_symptoms)
    for idx, symptom_dict in enumerate(df["Symptom Data"]):
        cycle_symptoms = defaultdict(list)
        for date_symptoms in symptom_dict.values():
            for symptom, severity in date_symptoms.items():
                cycle_symptoms[symptom].append(severity)
        for symptom in all_symptoms:
            symptom_matrix.at[idx, symptom] = (
                np.mean(cycle_symptoms[symptom]) if cycle_symptoms[symptom] else np.nan
            )
    fig = go.Figure(
        data=go.Heatmap(
            z=symptom_matrix.values,
            x=symptom_matrix.columns,
            y=symptom_matrix.index,
            colorscale="YlOrRd",
            zmin=1,
            zmax=10,
            hoverongaps=False,
        )
    )
    fig.update_layout(
        title="Average Symptom Severity Heatmap by Cycle",
        xaxis_title="Symptoms",
        yaxis_title="Cycle Index",
        xaxis_tickangle=-45,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig, use_container_width=True)
