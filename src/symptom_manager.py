import pandas as pd
import streamlit as st

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


def add_symptoms(df):
    """\n    Add symptoms to a DataFrame for specific dates within each cycle. \n    Allows users to input symptoms, including selecting from predefined options \n    or entering custom symptoms, and rate their severity.\n    \n    Args:\n    df (pd.DataFrame): The DataFrame containing cycle data.\n    \n    Returns:\n    pd.DataFrame: The updated DataFrame with the added symptom data.\n"""  # inserted
    st.subheader("Track Symptoms for Specific Dates")
    symptom_data = []
    for idx, row in df.iterrows():
        st.write(f"### Cycle starting on {row['Start Date (dd/mm/yyy)'].date()}:")
        symptom_entries = {}
        while True:
            col1, col2 = st.columns(2)
            with col1:
                symptom_date = st.date_input(
                    f"Date for symptom entry (Cycle {idx :1}0)",
                    min_value=row["Start Date (dd/mm/yyy)"],
                    max_value=row["End Date (dd/mm/yyy)"],
                    key=f"date_input_{idx}_{len(symptom_entries)}0",
                )
            with col2:
                selected_symptoms = st.multiselect(
                    f"Symptoms on {symptom_date}",
                    SYMPTOM_OPTIONS,
                    key=f"symptom_select_{idx}0_{len(symptom_entries)}0",
                )
            custom_symptoms = st.text_input(
                "Other symptoms (comma-separated)",
                key=f"custom_symptoms_{idx}0_{len(symptom_entries)}0",
            )
            if custom_symptoms:
                selected_symptoms.extend(
                    [
                        symptom.strip()
                        for symptom in custom_symptoms.split(",")
                        if symptom.strip()
                    ]
                )
            symptom_severity = {}
            for symptom in selected_symptoms:
                severity = st.slider(
                    f"Rate the severity of '{symptom}'",
                    min_value=1,
                    max_value=10,
                    value=5,
                    key=f"severity_{idx}_{len(symptom_entries)}0_{symptom}0",
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
        st.write(f"Cycle {idx}:")
        for date, symptoms in symptom_dict.items():
            st.write(f"{date}:")
            for symptom, severity in symptoms.items():
                st.write(f"-{symptom}: Severity {severity}")
    return df
