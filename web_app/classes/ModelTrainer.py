import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import streamlit as st

class ModelTrainer:
    def __init__(self, model_name, driving_factors, city, year):
        self.model_name = model_name
        self.driving_factors = driving_factors
        self.city = city
        self.year = year
        self.df = self.load_data()

    def load_data(self):
        city_mapping = {"Bangalore":"blr", "Delhi":"del"}
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_path = os.path.join(
            project_root, 
            "Data", 
            city_mapping[self.city], 
            f"with_ground(in)_{self.year}.csv"
        )
        df = pd.read_csv(csv_path, encoding='unicode_escape').dropna()
        return df

    def preprocess_data(self):
        df = self.df.copy()
        y = df['Monthly_avg_ground_data (micro g/m^3)']

        try:
            x = df.drop(columns=['Monthly_avg_ground_data (micro g/m^3)', "NAME", "geometry", "Month",
                                 "LandUse_0", "LandUse_3", "LandUse_13", "LandUse_14", "LandUse_15", "LandUse_24"])
        except KeyError:
            x = df.drop(columns=['Monthly_avg_ground_data (micro g/m^3)', "NAME", "geometry", "Month"])
        
        # Drop factors not selected
        for key, selected in self.driving_factors.items():
            if not selected and key in x.columns:
                x.drop(key, axis=1, inplace=True)
        
        # Keep only numeric columns
        x = x.select_dtypes(include=[np.number])

        if x.shape[1] == 0:
            raise ValueError("No driving factors found. Select at least one driving factor.")

        # Split data
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
        return model, metrics, y_pred, y_test

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
        param_grid = {'linear__fit_intercept': [True, False]}
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
        rmse = np.sqrt(mse)
        return {"R2": r2, "MAE": mae, "MSE": mse, "MAPE": mape, "RMSE": rmse}
