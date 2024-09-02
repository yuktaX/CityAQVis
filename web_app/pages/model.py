import streamlit as st
import ee
import geemap.foliumap as geemap
import geopandas as gpd  # used to read the shapfile
import rasterio as rio   # used to read the raster (.tif) files
from rasterio.plot import show # used to make plots using rasterio
import matplotlib.pyplot as plt #to make plots using matplotlib
import seaborn as sns
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR


# Define a function to extract raster values for a given point from a TIFF file
def extract_raster_values(point, raster_file):
    longitude = point['geometry'].x
    latitude = point['geometry'].y
    with rio.open(raster_file) as src:
        row, col = src.index(longitude, latitude)
        value = src.read(1)[row, col]  # Assuming single band raster
    return value

st.set_page_config(layout="wide")

st.title("Build your model")

col1, col2 = st.columns(2)

def trainModel(selected_model, driving_factors):
    df = pd.read_csv("/home/brij/studies/sem7/airPollution/Project_Elective/web_app/DownloadTest/DownloadTest/with_ground(in).csv", encoding = 'unicode_escape')
    df = df.dropna()

    # Reshape the input data to be 2D arrays
    y = df['Monthly_avg_ground_data (µg/m³)']
    x = df.drop(columns=['Monthly_avg_ground_data (µg/m³)',"NAME", "geometry", "Month", "LandUse_0", "LandUse_3", "LandUse_13", "LandUse_14", "LandUse_15", "LandUse_24"])


    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, random_state = 31)

    #defining function for evaluation
    def mean_absolute_percentage_error(y_true, y_pred):
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    #Gradient Boosting regressor with hyperparameter tuning using GridSearch

    np.random.seed(55)

    
    
    if selected_model == "GBR":
        # Define the parameter grid
        param_grid = {
            'n_estimators': [25, 50, 100, 200],
            'learning_rate': [0.1, 0.01, 0.001],
            'max_depth': [3, 5, 8, 12, 16]
        }
        # Initialize the Gradient Boosting Regressor
        model = GradientBoostingRegressor()
        
        # Create the GridSearchCV object
        grid_search = GridSearchCV(model, param_grid, cv=2)

        # Train the model with hyperparameter tuning
        grid_search.fit(x_train, y_train)

        # Get the best model with tuned hyperparameters
        best_model = grid_search.best_estimator_

        # Make predictions using the best model
        y_pred = best_model.predict(x_test)

    elif selected_model == "Linear regression":
        # Define the parameter grid for Linear Regression
        param_grid = {
            'linear__fit_intercept': [True, False]
        }

        # Create a pipeline that includes scaling and the linear regression model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # You can remove this if you don't want to scale features
            ('linear', LinearRegression())
        ])

        # Create the GridSearchCV object
        grid_search = GridSearchCV(pipeline, param_grid, cv=2)

        # Train the model with hyperparameter tuning
        grid_search.fit(x_train, y_train)

        # Get the best model with tuned hyperparameters
        best_model = grid_search.best_estimator_

        # Make predictions using the best model
        y_pred = best_model.predict(x_test)
    elif selected_model == "SVM":
        # Define the parameter grid for SVR
        param_grid = {
            'svr__C': [0.1, 1, 10, 100],
            'svr__epsilon': [0.01, 0.1, 0.2],
            'svr__kernel': ['linear', 'rbf', 'poly'],
            'svr__gamma': ['scale', 'auto']  # Only relevant for 'rbf' and 'poly' kernels
        }

        # Create a pipeline that includes scaling and the SVR model
        pipeline = Pipeline([
            ('scaler', StandardScaler()),  # SVR performs better with scaled features
            ('svr', SVR())
        ])

        # Create the GridSearchCV object
        grid_search = GridSearchCV(pipeline, param_grid, cv=2)

        # Train the model with hyperparameter tuning
        grid_search.fit(x_train, y_train)

        # Get the best model with tuned hyperparameters
        best_model = grid_search.best_estimator_

        # Make predictions using the best model
        y_pred = best_model.predict(x_test)
    elif selected_model == "Random Forest":
        # Define the parameter grid
        param_grid = {
            'n_estimators': [15, 25, 50, 75, 100],
            'max_depth': [4, 6, 8],
            'min_samples_split': [2, 5, 10, 12],
            'min_samples_leaf': [1, 2, 4]
        }

        # Initialize the Random Forest Regressor
        rfr = RandomForestRegressor()

        # Create the GridSearchCV object
        grid_search = GridSearchCV(rfr, param_grid, cv=2)

        # Train the model with hyperparameter tuning
        grid_search.fit(x_train, y_train)

        # Get the best model with tuned hyperparameters
        best_model = grid_search.best_estimator_

        # Make predictions using the best model
        y_pred = best_model.predict(x_test)


    # Calculate the R^2 score
    r2 = r2_score(y_test, y_pred)
    print("R^2 Score:", r2)

    mae = mean_absolute_error(y_test, y_pred)
    print("Mean Absolute Error (MAE):", mae)

    mse = mean_squared_error(y_test, y_pred)
    print("Mean Squared Error (MSE):", mse)

    mape = mean_absolute_percentage_error(y_test, y_pred)
    print("Mean Absolute Percentage Error (MAPE):", mape)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print("Root Mean Squared Error (RMSE):" , rmse)

    return best_model, r2, mae, mse, mape, rmse



with col1:
    models = ["Linear regression", "SVM", "Random Forest", "GBR"]
    selected_model= st.selectbox("Model:", models)

    pollutants = ["NO2", "SO2", "CH4"]
    selected_pollutant= st.selectbox("Pollutant:", pollutants)

    st.write("Driving Factors")
    tropomi = st.checkbox("TROPOMI")
    elevation = st.checkbox("Elevation")
    rainfall = st.checkbox("Rainfall")
    population = st.checkbox("Popluation")
    viirs = st.checkbox("Night-time lights")
    temperature = st.checkbox("Temperature")
    windSpeed = st.checkbox("Wind Speed")

    best_model, r2, mae, mse, mape, rmse = trainModel(selected_model, [tropomi, elevation, rainfall, population, viirs, temperature, windSpeed])
    st.write("R2 score: ", r2)
    st.write("Mean Absolute Error (MAE):", mae)
    st.write("Mean Squared Error (MSE):", mse)
    st.write("Mean Absolute Percentage Error (MAPE):", mape)
    st.write("Root Mean Squared Error (RMSE):" , rmse)

    
with col2:
    
    st.write("train and show results")