import streamlit as st
import pandas as pd
from src.date_parser import convert_to_datetime
from src.cycle_calculator import calculate_cycle_data_v2
from src.symptom_manager import add_symptoms
from src.cycle_visualizer import visualize_symptom_frequency_and_severity
from src.cycle_visualizer import visualize_cycle_length
from src.cycle_visualizer import visualize_symptom_heatmap


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
            ) = calculate_cycle_data_v2(df)
            cycle_df = add_symptoms(cycle_df)

            st.subheader("Processed Data")
            st.dataframe(cycle_df)

            st.subheader("Cycle Insights")
            st.write(f"`Average Cycle Length:` **{avg_cycle_length:.2f}** days")
            st.write(
                f"`Standard Deviation of Cycle Length:` **{std_cycle_length:.2f}** days"
            )

            if next_cycle_prediction:
                st.write(f"`Next Cycle Prediction:` **{next_cycle_prediction.date()}**")
                st.write(
                    f"`95% Prediction Interval:` **{prediction_interval[0].date()}** to **{prediction_interval[1].date()}**"
                )
                st.write(
                    f"`Coefficient of Variation:` **{coefficient_of_variation:.2f}**%"
                )

                st.write("### Interpretation of Results:")
                st.write(
                    f"We are 95% confident that the next cycle will start between **{prediction_interval[0].date()} and {prediction_interval[1].date()}**."
                )

                if coefficient_of_variation < 10:
                    st.write(
                        f"Your cycles are considered regular your (`Coefficient of Variation`: {coefficient_of_variation} is ***less than*** 10%)."
                    )
                elif 10 <= coefficient_of_variation < 15:
                    st.write(
                        f"Your cycles show some variability (your `Coefficient of Variation`: {round(coefficient_of_variation, 2)} is ***greater than*** 10% but ***less than*** 15%)."
                    )
                else:
                    st.write(
                        "Your cycles show high variability (your `Coefficient of Variation`: {coefficient_of_variation} is ***greater than*** 15%). This may affect the accuracy of predictions."
                    )
            else:
                st.write(
                    "*Not enough data to make predictions or calculate confidence measures.*"
                )

            visualize_symptom_frequency_and_severity(cycle_df)
            visualize_symptom_heatmap(cycle_df)
            visualize_cycle_length(cycle_df)

        except Exception as e:
            st.error(f"Error processing the file: {e}")


if __name__ == "__main__":
    main()
