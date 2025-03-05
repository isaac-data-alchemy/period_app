# Period Application
* streamlit application for processing period data using only statistical methods and visualizing said data using **Plotly**, no large language models are utilized. Bare in mind the simplicity is intentional I made this for my partner. Also why there is no internal data store with the xecption of **cached data** for  speedy reruns.

# Project Set-up
* `git clone` https://github.com/isaac-data-alchemy/period_app.git
* `cd period_app`
* `run pip install -r requirements.txt`

## Running the Application locally
* `streamlit run app_v1.py`


## Visualization 
* `visualize_symptom_frequency_and_severity`: as the name suggests provides you with a plot of your symptoms colored by their frequency and severity.

* `visualize_symptom_heatmap`: a heat map that shows what symptoms rank most severe by color

*  `visualize_cycle_length`: plots a trendline of the lenght of your cycles, can help with observing trends in your cycle and variation in general.

