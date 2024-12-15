import streamlit as st
import folium
from folium.plugins import HeatMap
import streamlit.components.v1 as components
import plotly.express as px
import matplotlib.pyplot as plt


from classes.ModelTrainer import ModelTrainer
from classes.Visualizer import Visualiser
    

class App:
    def __init__(self):
        st.set_page_config(layout="wide")
        st.title("Build & Compare your models")

    def render(self):
        # Initialize session state variables if they don't exist
        if "metrics_col1" not in st.session_state:
            st.session_state["metrics_col1"] = None
        if "metrics_col2" not in st.session_state:
            st.session_state["metrics_col2"] = None
        if "viz_col1" not in st.session_state:
            st.session_state["viz_col1"] = None
        if "viz_col2" not in st.session_state:
            st.session_state["viz_col2"] = None
        if "ypred_col1" not in st.session_state:
            st.session_state["ypred_col1"] = None
        if "ytest_col1" not in st.session_state:
            st.session_state["ytest_col1"] = None
        if "ytest_col2" not in st.session_state:
            st.session_state["ytest_col2"] = None
        if "ypred_col2" not in st.session_state:
            st.session_state["ypred_col2"] = None

        col1, col2 = st.columns(2)

        # Column 1
        with col1:
            city = ["Bangalore", "Delhi"]
            selected_city = st.selectbox("City:", city, key="city_col1")
            
            models = ["Linear regression", "SVM", "Random Forest", "GBR"]
            selected_model = st.selectbox("Model:", models, key="model_col1")

            pollutants = ["NO2", "SO2", "CH4"]
            selected_pollutant = st.selectbox("Pollutant:", pollutants, key="pollutant_col1")
            
            if selected_city == "Delhi":
                year_of_study = ["2019"]
                selected_year = st.selectbox("Year Of Study: ", year_of_study, key="year_col1")
            else:
                year_of_study = ["2019", "2022"]
                selected_year = st.selectbox("Year Of Study: ", year_of_study, key="year_col1")

            # Streamlit Multiselect Dropdown
           
            selected_factors = st.multiselect(
                "Select Driving Factors",
                options=["NO2 (mol/m^2)", "Elevation", "Rainfall", "Population", "VIIRS", "Temperature", "WindSpeed"],
                default=[],
                key="driving_factors_col1"
            )

            if st.button("Train model", key="train_model_col1"):
                # Map selected factors to driving factor keys
                driving_factors = {
                    "NO2 (mol/m^2)": "NO2 (mol/m^2)" in selected_factors,
                    "Elevation": "Elevation" in selected_factors,
                    "Rainfall": "Rainfall" in selected_factors,
                    "Population": "Population" in selected_factors,
                    "VIIRS": "VIIRS" in selected_factors,
                    "Temperature": "Temperature" in selected_factors,
                    "WindSpeed": "WindSpeed" in selected_factors,
                }
                
                # Initialize the trainer and train the model
                trainer = ModelTrainer(selected_model, driving_factors, selected_city, year_of_study)
                model, metrics, y_pred, y_test = trainer.train_model()
                
                # Store results in session state
                st.session_state["metrics_col1"] = metrics
                st.session_state["ypred_col1"] = y_pred
                st.session_state["ytest_col1"] = y_test
                viz = Visualiser(model, driving_factors, selected_city)
                st.session_state["viz_col1"] = viz

            # Display results of model 1 if available
            if st.session_state["metrics_col1"]:
                st.write("### Model 1 Results:")
                st.write("R2 score: ", st.session_state["metrics_col1"]["R2"])
                st.write("Mean Absolute Error (MAE):", st.session_state["metrics_col1"]["MAE"])
                st.write("Mean Squared Error (MSE):", st.session_state["metrics_col1"]["MSE"])
                st.write("Mean Absolute Percentage Error (MAPE):", st.session_state["metrics_col1"]["MAPE"])
                st.write("Root Mean Squared Error (RMSE):", st.session_state["metrics_col1"]["RMSE"])

            if st.session_state["viz_col1"]:
                map_html_1 = st.session_state["viz_col1"].foliumMap()
                map_html_2 = st.session_state["viz_col1"].plotlyMap()

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(st.session_state["ytest_col1"], st.session_state["ypred_col1"], color='blue', label='Predicted vs Actual')
                ax.plot([st.session_state["ytest_col1"].min(), st.session_state["ytest_col1"].max()], [st.session_state["ytest_col1"].min(), st.session_state["ytest_col1"].max()], 'r--', lw=2, label='Ideal Fit')
                ax.set_title("Actual vs Predicted")
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.legend()
                st.pyplot(fig)

                # Display the map in Streamlit using st.components.v1.html()
                st.components.v1.html(map_html_1, height=400)
                st.plotly_chart(map_html_2)
                
        # Column 2
        with col2:
            city = ["Bangalore", "Delhi"]
            selected_city = st.selectbox("City:", city, key="city_col2")
            
            models = ["Linear regression", "SVM", "Random Forest", "GBR"]
            selected_model = st.selectbox("Model:", models, key="model_col2")

            pollutants = ["NO2", "SO2", "CH4"]
            selected_pollutant = st.selectbox("Pollutant:", pollutants, key="pollutant_col2")
            
            if selected_city == "Delhi":
                year_of_study = ["2019"]
                selected_year = st.selectbox("Year Of Study: ", year_of_study, key="year_col2")
            else:
                year_of_study = ["2019", "2022"]
                selected_year = st.selectbox("Year Of Study: ", year_of_study, key="year_col2")


            selected_factors = st.multiselect(
                "Select Driving Factors",
                options=["NO2 (mol/m^2)", "Elevation", "Rainfall", "Population", "VIIRS", "Temperature", "WindSpeed"],
                default=[],
                key="driving_factors_col2"
            )

            if st.button("Train model", key="train_model_col2"):
                # Map selected factors to driving factor keys
                driving_factors = {
                    "NO2 (mol/m^2)": "NO2 (mol/m^2)" in selected_factors,
                    "Elevation": "Elevation" in selected_factors,
                    "Rainfall": "Rainfall" in selected_factors,
                    "Population": "Population" in selected_factors,
                    "VIIRS": "VIIRS" in selected_factors,
                    "Temperature": "Temperature" in selected_factors,
                    "WindSpeed": "WindSpeed" in selected_factors,
                }
                
                trainer = ModelTrainer(selected_model, driving_factors, selected_city, year_of_study)
                model, metrics, y_pred, y_test = trainer.train_model()

                # Store results in session state
                st.session_state["metrics_col2"] = metrics
                st.session_state["ypred_col2"] = y_pred
                st.session_state["ytest_col2"] = y_test
                viz = Visualiser(model, driving_factors, selected_city)
                st.session_state["viz_col2"] = viz

            # Display results of model 2 if available
            if st.session_state["metrics_col2"]:
                st.write("### Model 2 Results:")
                st.write("R2 score: ", st.session_state["metrics_col2"]["R2"])
                st.write("Mean Absolute Error (MAE):", st.session_state["metrics_col2"]["MAE"])
                st.write("Mean Squared Error (MSE):", st.session_state["metrics_col2"]["MSE"])
                st.write("Mean Absolute Percentage Error (MAPE):", st.session_state["metrics_col2"]["MAPE"])
                st.write("Root Mean Squared Error (RMSE):", st.session_state["metrics_col2"]["RMSE"])
            
            if st.session_state["viz_col2"]:
                map_html_1 = st.session_state["viz_col2"].foliumMap()
                map_html_2 = st.session_state["viz_col2"].plotlyMap()

                fig, ax = plt.subplots(figsize=(8, 6))
                ax.scatter(st.session_state["ytest_col2"], st.session_state["ypred_col2"], color='blue', label='Predicted vs Actual')
                ax.plot([st.session_state["ytest_col2"].min(), st.session_state["ytest_col2"].max()], [st.session_state["ytest_col2"].min(), st.session_state["ytest_col2"].max()], 'r--', lw=2, label='Ideal Fit')
                ax.set_title("Actual vs Predicted")
                ax.set_xlabel("Actual Values")
                ax.set_ylabel("Predicted Values")
                ax.legend()
                st.pyplot(fig)

                # Display the map in Streamlit using st.components.v1.html()
                st.components.v1.html(map_html_1, height=400)
                st.plotly_chart(map_html_2)

            
if __name__ == "__main__":
    app = App()
    app.render()
