# CityAQVis: Machine Learning Sandbox With Comparative Visual Analytics For Air Quality In Urban Regions Using Multi-Source Data Bangalore Air Pollution Susceptibility Maps

## Part 1: Setting up the Dataset, Colab Notebook and training the ML model

- `CityAQVis_Dataset_ML_Model.ipynb` contains the code for the project
- `Drive` folder is the folder which contains all the raster files downloaded from GEE. This is mainly Bangalore 2019 data.
- `dataset.csv` denotes the final dataset we obtained and trained the model on for Bangalore 2019.

## Part 2: Creating a web-app to visualize and interact with the data
We have used streamlit to build this application.
### Setup
1. Install streamlit on your system. Some of the dependent libraries that streamlit uses have outdated tools so please check which version works with all the other python libraries it uses. This is mentioned in `requirements.txt`
2. Clone `webapp` locally and in the terminal run `streamlit run app.py`
3. The landing page has more instructions and information on how to use the application. You can refer to this to use it and modify it for your own usecase.

   
![webapp-1](https://github.com/user-attachments/assets/a6fed05e-85f7-416e-910f-b0dcf55cb958)

![app-1](https://github.com/user-attachments/assets/cdd16873-55e7-4c82-b251-f66c81d5274e)
![app-2](https://github.com/user-attachments/assets/e749883d-a249-47cb-b02c-ab7b0c8c6317)
![app-3](https://github.com/user-attachments/assets/79c5c2e9-63ca-42d8-a71b-daffde803595)
