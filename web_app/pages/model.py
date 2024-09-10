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


class ModelTrainer:
    def __init__(self, model_name, driving_factors):
        self.model_name = model_name
        self.driving_factors = driving_factors
        self.df = self.load_data()

    def load_data(self):
        df = pd.read_csv("/home/brij/studies/sem7/airPollution/Project_Elective/web_app/DownloadTest/DownloadTest/with_ground(in).csv", encoding='unicode_escape')
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


class App:
    def __init__(self):
        st.set_page_config(layout="wide")
        st.title("Build your model")

    def render(self):
        col1, col2 = st.columns(2)

        with col1:
            models = ["Linear regression", "SVM", "Random Forest", "GBR"]
            selected_model = st.selectbox("Model:", models)

            pollutants = ["NO2", "SO2", "CH4"]
            selected_pollutant = st.selectbox("Pollutant:", pollutants)

            st.write("Driving Factors")
            tropomi = st.checkbox("TROPOMI")
            elevation = st.checkbox("Elevation")
            rainfall = st.checkbox("Rainfall")
            population = st.checkbox("Population")
            viirs = st.checkbox("Night-time lights")
            temperature = st.checkbox("Temperature")
            wind_speed = st.checkbox("Wind Speed")

            if st.button("Train model"):
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

                st.write("R2 score: ", metrics["R2"])
                st.write("Mean Absolute Error (MAE):", metrics["MAE"])
                st.write("Mean Squared Error (MSE):", metrics["MSE"])
                st.write("Mean Absolute Percentage Error (MAPE):", metrics["MAPE"])
                st.write("Root Mean Squared Error (RMSE):", metrics["RMSE"])

        with col2:
            st.write("Train and show results")


if __name__ == "__main__":
    app = App()
    app.render()
