import streamlit as st
import pytest
import pandas as pd
import numpy as np
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from collections import defaultdict


# Predefined list of common symptoms
SYMPTOM_OPTIONS = [
    "Body Pain",
    "Head Hurting",
    "Dizziness",
    "Emotional Dysregulation",
    "Leg Pains",
    "Cramps",
    "Stomach Pain",
    "Headache",
    "Bloating",
    "Fatigue",
    "Breast Tenderness",
    "Mood Swings",
    "Anxiety",
    "Insomnia",
    "Cravings",
    "Back Pain",
]


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
        st.error(f"Error in converting '{column_name}': {e}")


def calculate_cycle_data(df):
    """
    Calculate various menstrual cycle data based on the provided DataFrame.

    Parameters:
    - df: DataFrame containing 'Start Date (dd/mm/yyy)' and 'End Date (dd/mm/yyy)' columns.

    Returns:
    - df: Updated DataFrame with additional columns for cycle data.
    - avg_cycle_length: Average length of menstrual cycles.
    - std_cycle_length: Standard deviation of menstrual cycle lengths.
    - next_cycle_prediction: Predicted start date of the next cycle.
    - prediction_interval: 95% prediction interval for the next cycle start date.
    - coefficient_of_variation: Coefficient of variation for menstrual cycle lengths.

    Raises:
    - st.error: If an error occurs during the calculation process.
    """
    try:
        df["Period Duration (days)"] = (
            df["End Date (dd/mm/yyy)"] - df["Start Date (dd/mm/yyy)"]
        ).dt.days
        df["Cycle Duration (days)"] = df["Start Date (dd/mm/yyy)"].diff().dt.days.abs()
        cycle_durations = df["Cycle Duration (days)"].dropna()
        df["Cycle Regularity (days)"] = cycle_durations.diff().fillna(0).abs()

        avg_cycle_length = cycle_durations.mean()
        std_cycle_length = cycle_durations.std()

        next_cycle_prediction = None
        prediction_interval = None
        coefficient_of_variation = None

        if pd.notnull(avg_cycle_length) and len(cycle_durations) > 1:
            last_cycle_start = df["Start Date (dd/mm/yyy)"].iloc[-1]
            next_cycle_prediction = last_cycle_start + timedelta(days=avg_cycle_length)

            # Calculate standard error of prediction
            se_prediction = std_cycle_length * np.sqrt(1 + 1 / len(cycle_durations))

            # Calculate 95% prediction interval
            t_value = stats.t.ppf(0.975, df=len(cycle_durations) - 1)
            margin_of_error = t_value * se_prediction
            prediction_interval = (
                next_cycle_prediction - timedelta(days=margin_of_error),
                next_cycle_prediction + timedelta(days=margin_of_error),
            )

            # Calculate coefficient of variation
            coefficient_of_variation = (std_cycle_length / avg_cycle_length) * 100

        return (
            df,
            avg_cycle_length,
            std_cycle_length,
            next_cycle_prediction,
            prediction_interval,
            coefficient_of_variation,
        )
    except Exception as e:
        st.error(f"Error in calculating cycle data: {e}")


def add_symptoms(df):
    """
    Add symptoms to a DataFrame for specific dates within each cycle.
    Allows users to input symptoms, including selecting from predefined options
    or entering custom symptoms, and rate their severity.

    Args:
    df (pd.DataFrame): The DataFrame containing cycle data.

    Returns:
    pd.DataFrame: The updated DataFrame with the added symptom data.
    """
    st.subheader("Track Symptoms for Specific Dates")

    symptom_data = []

    for idx, row in df.iterrows():
        st.write(f"### Cycle starting on {row['Start Date (dd/mm/yyy)'].date()}:")

        symptom_entries = {}

        while True:
            col1, col2 = st.columns(2)

            with col1:
                symptom_date = st.date_input(
                    f"Date for symptom entry (Cycle {idx + 1})",
                    min_value=row["Start Date (dd/mm/yyy)"],
                    max_value=row["End Date (dd/mm/yyy)"],
                    key=f"date_input_{idx}_{len(symptom_entries)}",
                )

            with col2:
                selected_symptoms = st.multiselect(
                    f"Symptoms on {symptom_date}",
                    SYMPTOM_OPTIONS,
                    key=f"symptom_select_{idx}_{len(symptom_entries)}",
                )

            custom_symptoms = st.text_input(
                f"Other symptoms (comma-separated)",
                key=f"custom_symptoms_{idx}_{len(symptom_entries)}",
            )

            if custom_symptoms:
                selected_symptoms.extend(
                    [
                        symptom.strip()
                        for symptom in custom_symptoms.split(",")
                        if symptom.strip()
                    ]
                )

            # New: Add severity ratings for each symptom
            symptom_severity = {}
            for symptom in selected_symptoms:
                severity = st.slider(
                    f"Rate the severity of '{symptom}'",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key=f"severity_{idx}_{len(symptom_entries)}_{symptom}",
                )
                symptom_severity[symptom] = severity

            symptom_entries[symptom_date] = symptom_severity

            if not st.checkbox(
                "Add another entry for this cycle?",
                key=f"add_entry_{idx}_{len(symptom_entries)}",
            ):
                break

        symptom_data.append(symptom_entries)

    df["Symptom Data"] = symptom_data

    st.write("### Symptoms Added with Dates and Severity:")
    for idx, symptom_dict in enumerate(df["Symptom Data"]):
        st.write(f"Cycle {idx + 1}:")
        for date, symptoms in symptom_dict.items():
            st.write(f"  {date}:")
            for symptom, severity in symptoms.items():
                st.write(f"    - {symptom}: Severity {severity}")

    return df


def visualize_symptom_frequency_and_severity(df):
    """
    Visualizes the frequency and average severity of symptoms across menstrual cycles using Plotly.

    Parameters:
    - df (pandas.DataFrame): DataFrame containing the 'Symptom Data' column with symptom information.

    Returns:
    - None
    """
    symptom_data = defaultdict(lambda: {"count": 0, "total_severity": 0})

    for symptom_dict in df["Symptom Data"]:
        for date_symptoms in symptom_dict.values():
            for symptom, severity in date_symptoms.items():
                symptom_data[symptom]["count"] += 1
                symptom_data[symptom]["total_severity"] += severity

    symptom_df = pd.DataFrame(
        [
            {
                "Symptom": symptom,
                "Count": data["count"],
                "Avg Severity": data["total_severity"] / data["count"],
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
    all_symptoms = sorted(
        set(
            sym
            for symptom_dict in df["Symptom Data"]
            for date_symptoms in symptom_dict.values()
            for sym in date_symptoms.keys()
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


def pattern_recognition(df):
    """
    Performs basic pattern recognition on symptom data.

    Parameters:
    - df: DataFrame containing 'Symptom Data' column with symptom information for each cycle.

    Returns:
    - dict: A dictionary containing recognized patterns.
    """
    patterns = {
        "most_common_symptoms": [],
        "most_severe_symptoms": [],
        "symptom_correlations": [],
    }

    all_symptoms = defaultdict(lambda: {"count": 0, "total_severity": 0})

    for symptom_dict in df["Symptom Data"]:
        cycle_symptoms = set()
        for date_symptoms in symptom_dict.values():
            for symptom, severity in date_symptoms.items():
                all_symptoms[symptom]["count"] += 1
                all_symptoms[symptom]["total_severity"] += severity
                cycle_symptoms.add(symptom)

        # Check for symptom correlations within each cycle
        for sym1 in cycle_symptoms:
            for sym2 in cycle_symptoms:
                if sym1 < sym2:  # To avoid duplicates
                    patterns["symptom_correlations"].append((sym1, sym2))

    # Find most common symptoms
    patterns["most_common_symptoms"] = sorted(
        all_symptoms.items(), key=lambda x: x[1]["count"], reverse=True
    )[:5]

    # Find most severe symptoms
    patterns["most_severe_symptoms"] = sorted(
        all_symptoms.items(),
        key=lambda x: x[1]["total_severity"] / x[1]["count"],
        reverse=True,
    )[:5]

    # Count symptom correlations
    correlation_counts = defaultdict(int)
    for pair in patterns["symptom_correlations"]:
        correlation_counts[pair] += 1
    patterns["symptom_correlations"] = sorted(
        correlation_counts.items(), key=lambda x: x[1], reverse=True
    )[:5]

    return patterns


def simple_knowledge_base():
    """
    Creates a simple knowledge base for symptom recommendations.

    Returns:
    - dict: A dictionary containing recommendations for various symptoms.
    """
    return {
        "Cramps": [
            "Apply a heating pad to your lower abdomen",
            "Practice gentle yoga or stretching exercises",
            "Stay hydrated and avoid caffeine",
        ],
        "Headache": [
            "Rest in a dark, quiet room",
            "Try cold or warm compresses on your forehead",
            "Stay hydrated and consider magnesium-rich foods",
        ],
        "Fatigue": [
            "Ensure you're getting enough sleep",
            "Engage in light exercise like walking",
            "Consider iron-rich foods or supplements (consult your doctor)",
        ],
        "Mood Swings": [
            "Practice mindfulness or meditation",
            "Engage in regular exercise",
            "Ensure a balanced diet with omega-3 fatty acids",
        ],
        "Bloating": [
            "Avoid salty foods",
            "Stay hydrated and consider herbal teas like peppermint",
            "Engage in light exercise to reduce water retention",
        ],
    }


def generate_recommendations(patterns, knowledge_base):
    """
    Generates personalized recommendations based on recognized patterns and knowledge base.

    Parameters:
    - patterns: dict containing recognized symptom patterns
    - knowledge_base: dict containing recommendations for various symptoms

    Returns:
    - list: A list of personalized recommendations
    """
    recommendations = []

    for symptom, data in patterns["most_common_symptoms"][:3]:
        if symptom in knowledge_base:
            recommendations.extend(knowledge_base[symptom])

    for symptom, data in patterns["most_severe_symptoms"][:3]:
        if (
            symptom in knowledge_base
            and knowledge_base[symptom][0] not in recommendations
        ):
            recommendations.append(knowledge_base[symptom][0])

    return list(set(recommendations))  # Remove duplicates


def display_insights_and_recommendations(df):
    """
    Displays insights from pattern recognition and generates recommendations.

    Parameters:
    - df: DataFrame containing 'Symptom Data' column with symptom information for each cycle.

    Returns:
    - None
    """
    patterns = pattern_recognition(df)
    kb = simple_knowledge_base()
    recommendations = generate_recommendations(patterns, kb)

    st.subheader("Symptom Insights")

    st.write("Most Common Symptoms:")
    for symptom, data in patterns["most_common_symptoms"]:
        st.write(f"- {symptom} (Occurred {data['count']} times)")

    st.write("\nMost Severe Symptoms (On Average):")
    for symptom, data in patterns["most_severe_symptoms"]:
        avg_severity = data["total_severity"] / data["count"]
        st.write(f"- {symptom} (Average severity: {avg_severity:.2f})")

    st.write("\nCommon Symptom Pairs:")
    for (sym1, sym2), count in patterns["symptom_correlations"]:
        st.write(f"- {sym1} often occurs with {sym2} ({count} times)")

    st.subheader("Personalized Recommendations")
    for rec in recommendations:
        st.write(f"- {rec}")

    st.write(
        "\nPlease note: These recommendations are based on general advice and your recorded symptoms. Always consult with a healthcare professional for medical advice."
    )


def main():
    st.title(
        "Menstrual Cycle Tracker with Date-Specific Symptom Tracking and Confidence Measures"
    )

    uploaded_file = st.file_uploader(
        "Upload your menstrual cycle data (Excel file)", type=["xlsx"]
    )

    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file)
            convert_to_datetime(df, "Start Date (dd/mm/yyy)")
            convert_to_datetime(df, "End Date (dd/mm/yyy)")

            (
                cycle_df,
                avg_cycle_length,
                std_cycle_length,
                next_cycle_prediction,
                prediction_interval,
                coefficient_of_variation,
            ) = calculate_cycle_data(df)
            cycle_df = add_symptoms(cycle_df)

            st.subheader("Processed Data")
            st.dataframe(cycle_df)

            st.subheader("Cycle Insights")
            st.write(f"**Average Cycle Length:** {avg_cycle_length:.2f} days")
            st.write(
                f"**Standard Deviation of Cycle Length:** {std_cycle_length:.2f} days"
            )

            if next_cycle_prediction:
                st.write(f"**Next Cycle Prediction:** {next_cycle_prediction.date()}")
                st.write(
                    f"**95% Prediction Interval:** {prediction_interval[0].date()} to {prediction_interval[1].date()}"
                )
                st.write(
                    f"**Coefficient of Variation:** {coefficient_of_variation:.2f}%"
                )

                st.write("### Interpretation of Results:")
                st.write(
                    f"We are 95% confident that the next cycle will start between {prediction_interval[0].date()} and {prediction_interval[1].date()}."
                )

                if coefficient_of_variation < 10:
                    st.write(
                        "Your cycles are considered regular (Coefficient of Variation < 10%)."
                    )
                elif 10 <= coefficient_of_variation < 15:
                    st.write(
                        "Your cycles show some variability (10% ≤ Coefficient of Variation < 15%)."
                    )
                else:
                    st.write(
                        "Your cycles show high variability (Coefficient of Variation ≥ 15%). This may affect the accuracy of predictions."
                    )
            else:
                st.write(
                    "Not enough data to make predictions or calculate confidence measures."
                )

            visualize_symptom_frequency_and_severity(cycle_df)
            visualize_symptom_heatmap(cycle_df)
            visualize_cycle_length(cycle_df)

        except Exception as e:
            st.error(f"Error processing the file: {e}")


if __name__ == "__main__":
    main()
