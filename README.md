# CityAQVis: Machine Learning Sandbox With Comparative Visual Analytics For Air Quality In Urban Regions Using Multi-Source Data Bangalore Air Pollution Susceptibility Maps

## Part 1: Setting up the Dataset, Colab Notebook and training the ML model

- `CityAQVis_Dataset_ML_Model.ipynb` contains the code for the project
- `DownloadTest.zip` is the folder which contains all the raster files downloaded from GEE. This contains files for Bangalore 2019, 2022 and Dehli 2019.
- `dataset.csv` denotes the final dataset we obtained and trained the model on for Bangalore 2019.

## Part 2: Creating a web-app to visualize and interact with the data
We have used streamlit to build this application. We developed an interactive sandbox environment that enables users to train,
compare, and visualize NO2 predictions across urban areas. This tool allows for flexible experimentation with different machine learning models and datasets, making it adaptable to various cities and pollutants.

Refer to [webapp readme](/web_app/readme.md) for the detailed description and setup.
