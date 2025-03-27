# CityAQVis: Machine Learning Sandbox With Comparative Visual Analytics For Air Quality In Urban Regions Using Multi-Source Data Bangalore Air Pollution Susceptibility Maps

## Part 1: Setting up the Dataset, Colab Notebook and training the ML model

- `CityAQVis_Dataset_ML_Model.ipynb` contains the code for the project
- `DownloadTest.zip` is the folder which contains all the raster files downloaded from GEE. This contains files for Bangalore 2019, 2022 and Dehli 2019.
- `dataset.csv` denotes the final dataset we obtained and trained the model on for Bangalore 2019.

## Part 2: Creating a web-app to visualize and interact with the data
We have used streamlit to build this application. We developed an interactive sandbox environment that enables users to train,
compare, and visualize NO2 predictions across urban areas. This tool allows for flexible experimentation with different machine learning models and datasets, making it adaptable to various cities and pollutants.

Refer 
### Setup
1. Install streamlit on your system. Some of the dependent libraries that streamlit uses have outdated tools so please check which version works with all the other python libraries it uses. This is mentioned in `requirements.txt`
2. Clone `webapp` locally and in the terminal run `streamlit run app.py`
3. The readme.md of webapp has further instructions, refer to that.
4. The landing page has more instructions and information on how to use the application. You can refer to this to use it and modify it for your own usecase.

   
![webapp-1](https://github.com/user-attachments/assets/a6fed05e-85f7-416e-910f-b0dcf55cb958)
![webapp-2](https://github.com/user-attachments/assets/cbc13f53-98a4-43b2-8e99-d4f5f6081fb6)

![app-1](https://github.com/user-attachments/assets/cdd16873-55e7-4c82-b251-f66c81d5274e)
![app-2](https://github.com/user-attachments/assets/e749883d-a249-47cb-b02c-ab7b0c8c6317)
![app-3](https://github.com/user-attachments/assets/79c5c2e9-63ca-42d8-a71b-daffde803595)
