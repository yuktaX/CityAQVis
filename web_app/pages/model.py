import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import folium
from folium.plugins import HeatMap
import streamlit.components.v1 as components



class ModelTrainer:
    def __init__(self, model_name, driving_factors):
        self.model_name = model_name
        self.driving_factors = driving_factors
        self.df = self.load_data()

    def load_data(self):
        #df = pd.read_csv("/home/brij/studies/sem7/airPollution/Project_Elective/web_app/DownloadTest/DownloadTest/with_ground(in).csv", encoding='unicode_escape')
        df = pd.read_csv("/home/yukta/College/sem7/RE-Work-Jaya/Project_Elective_Sem6/web_app/DownloadTest/DownloadTest/with_ground(in).csv", encoding='unicode_escape')
        df = df.dropna()
        return df

    def preprocess_data(self):
        df = self.df.copy()

        y = df['Monthly_avg_ground_data (µg/m³)']
        x = df.drop(columns=['Monthly_avg_ground_data (µg/m³)', "NAME", "geometry", "Month", "LandUse_0", "LandUse_3", "LandUse_13", "LandUse_14", "LandUse_15", "LandUse_24"])

        # Drop factors not selected
        for key in self.driving_factors:
            if not self.driving_factors[key]:
                x.drop(key, axis=1, inplace=True)

        # Split data into train and test sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state=31)
        return x_train, x_test, y_train, y_test

    def mean_absolute_percentage_error(self, y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def train_model(self):
        x_train, x_test, y_train, y_test = self.preprocess_data()
        if self.model_name == "GBR":
            model, y_pred = self.gradient_boosting(x_train, y_train, x_test)
        elif self.model_name == "Linear regression":
            model, y_pred = self.linear_regression(x_train, y_train, x_test)
        elif self.model_name == "SVM":
            model, y_pred = self.svm(x_train, y_train, x_test)
        elif self.model_name == "Random Forest":
            model, y_pred = self.random_forest(x_train, y_train, x_test)
        else:
            raise ValueError("Unsupported model selected")

        metrics = self.evaluate_model(y_test, y_pred)
        return model, metrics

    def gradient_boosting(self, x_train, y_train, x_test):
        np.random.seed(55)

        param_grid = {
            'n_estimators': [25, 50, 100, 200],
            'learning_rate': [0.1, 0.01, 0.001],
            'max_depth': [3, 5, 8, 12, 16]
        }
        model = GradientBoostingRegressor()
        grid_search = GridSearchCV(model, param_grid, cv=2)
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(x_test)
        return best_model, y_pred

    def linear_regression(self, x_train, y_train, x_test):
        param_grid = {
            'linear__fit_intercept': [True, False]
        }

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('linear', LinearRegression())
        ])
        grid_search = GridSearchCV(pipeline, param_grid, cv=2)
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(x_test)
        return best_model, y_pred

    def svm(self, x_train, y_train, x_test):
        param_grid = {
            'svr__C': [0.1, 1, 10, 100],
            'svr__epsilon': [0.01, 0.1, 0.2],
            'svr__kernel': ['linear', 'rbf', 'poly'],
            'svr__gamma': ['scale', 'auto']
        }
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svr', SVR())
        ])
        grid_search = GridSearchCV(pipeline, param_grid, cv=2)
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(x_test)
        return best_model, y_pred

    def random_forest(self, x_train, y_train, x_test):
        param_grid = {
            'n_estimators': [15, 25, 50, 75, 100],
            'max_depth': [4, 6, 8],
            'min_samples_split': [2, 5, 10, 12],
            'min_samples_leaf': [1, 2, 4]
        }
        model = RandomForestRegressor()
        grid_search = GridSearchCV(model, param_grid, cv=2)
        grid_search.fit(x_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(x_test)
        return best_model, y_pred

    def evaluate_model(self, y_test, y_pred):
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mape = self.mean_absolute_percentage_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred, squared=False)
        return {"R2": r2, "MAE": mae, "MSE": mse, "MAPE": mape, "RMSE": rmse}
    

class Visualiser:
    def __init__(self, model, driving_factors) -> None:
        self.model = model
        self.driving_factors = driving_factors
        self.grid_df = pd.read_csv("blr.csv")
    
    def foliumMap(self):
        features = []
        for key in self.driving_factors:
            if self.driving_factors[key]:
                features.append(key)

        self.grid_df['NO2_prediction'] = self.model.predict(self.grid_df[features])

        # Create a base map centered around the city
        lat_min, lat_max = 12.85, 13.20
        lon_min, lon_max = 77.45, 77.80
    
        min_zoom, max_zoom = 1, 13

        # Create a base map centered around the city
        m = folium.Map(location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], zoom_start=12, min_zoom=min_zoom, max_zoom=max_zoom)

        # Convert predictions to a list of [latitude, longitude, NO2] for HeatMap
        heat_data = [[row['latitude'], row['longitude'], row['NO2_prediction']] for index, row in self.grid_df.iterrows()]

        # Add the heatmap layer with NO2 predictions
        HeatMap(heat_data, radius=20, blur=25, max_zoom=12, min_opacity=0.4).add_to(m)

        # HTML for the custom legend
        legend_html = '''
        <div style="
            position: fixed;
            bottom: 50px; left: 50px; width: 150px; height: 150px;
            background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
            padding: 0px;
            ">
            <b>NO2 Levels</b><br>
            <i style="background: rgba(0, 0, 255, 0.5);width: 20px;height: 10px;display: inline-block;"></i> Low (<10 μg/m³)<br>
            <i style="background: rgba(0, 255, 0, 0.5);width: 20px;height: 10px;display: inline-block;"></i> Moderate (10-20 μg/m³)<br>
            <i style="background: rgba(255, 255, 0, 0.5);width: 20px;height: 10px;display: inline-block;"></i> High (20-40 μg/m³)<br>
            <i style="background: rgba(255, 0, 0, 0.5);width: 20px;height: 10px;display: inline-block;"></i> Very High (>40 μg/m³)
        </div>
        '''

        # Add the custom legend to the map
        m.get_root().html.add_child(folium.Element(legend_html))

        # Save the map to an HTML file or display directly
        m.save('no2_heatmap_with_legend.html')
        map_html = m._repr_html_()
        return map_html


class App:
    def __init__(self):
        st.set_page_config(layout="wide")
        st.title("Build your models")

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

        col1, col2 = st.columns(2)

        # Column 1
        with col1:
            models = ["Linear regression", "SVM", "Random Forest", "GBR"]
            selected_model = st.selectbox("Model:", models, key="model_col1")

            pollutants = ["NO2", "SO2", "CH4"]
            selected_pollutant = st.selectbox("Pollutant:", pollutants, key="pollutant_col1")

            st.write("Driving Factors")
            tropomi = st.checkbox("TROPOMI", key="tropomi_col1")
            elevation = st.checkbox("Elevation", key="elevation_col1")
            rainfall = st.checkbox("Rainfall", key="rainfall_col1")
            population = st.checkbox("Population", key="population_col1")
            viirs = st.checkbox("Night-time lights", key="viirs_col1")
            temperature = st.checkbox("Temperature", key="temperature_col1")
            wind_speed = st.checkbox("Wind Speed", key="windspeed_col1")

            if st.button("Train model", key="train_model_col1"):
                driving_factors = {
                    "NO2 (mol/m^2)": tropomi,
                    "Elevation": elevation,
                    "Rainfall": rainfall,
                    "Population": population,
                    "VIIRS": viirs,
                    "Temperature": temperature,
                    "WindSpeed": wind_speed
                }
                trainer = ModelTrainer(selected_model, driving_factors)
                model, metrics = trainer.train_model()

                # Store results in session state
                st.session_state["metrics_col1"] = metrics
                viz = Visualiser(model, driving_factors)
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
                map_html = st.session_state["viz_col1"].foliumMap()

                # Display the map in Streamlit using st.components.v1.html()
                st.components.v1.html(map_html, height=800)

        # Column 2
        with col2:
            models = ["Linear regression", "SVM", "Random Forest", "GBR"]
            selected_model = st.selectbox("Model:", models, key="model_col2")

            pollutants = ["NO2", "SO2", "CH4"]
            selected_pollutant = st.selectbox("Pollutant:", pollutants, key="pollutant_col2")

            st.write("Driving Factors")
            tropomi = st.checkbox("TROPOMI", key="tropomi_col2")
            elevation = st.checkbox("Elevation", key="elevation_col2")
            rainfall = st.checkbox("Rainfall", key="rainfall_col2")
            population = st.checkbox("Population", key="population_col2")
            viirs = st.checkbox("Night-time lights", key="viirs_col2")
            temperature = st.checkbox("Temperature", key="temperature_col2")
            wind_speed = st.checkbox("Wind Speed", key="windspeed_col2")

            if st.button("Train model", key="train_model_col2"):
                driving_factors = {
                    "NO2 (mol/m^2)": tropomi,
                    "Elevation": elevation,
                    "Rainfall": rainfall,
                    "Population": population,
                    "VIIRS": viirs,
                    "Temperature": temperature,
                    "WindSpeed": wind_speed
                }
                trainer = ModelTrainer(selected_model, driving_factors)
                model, metrics = trainer.train_model()

                # Store results in session state
                st.session_state["metrics_col2"] = metrics
                viz = Visualiser(model, driving_factors)
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
                map_html = st.session_state["viz_col2"].foliumMap()

                # Display the map in Streamlit using st.components.v1.html()
                st.components.v1.html(map_html, height=600)

if __name__ == "__main__":
    app = App()
    app.render()
