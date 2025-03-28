# CityAQVis - Air Quality Visualization Platform  

This repository contains the code and data for CityAQVis, a machine learning-based platform for air quality prediction and visualization.  

## Project Structure  

- **`.streamlit/`** – Streamlit configuration files for customizing the web application settings.  
- **`.venv/`** – Virtual environment directory containing dependencies (not required for deployment).  
- **`Data/`** – Contains data files required for model training and visualization for each city. The relative file paths here are important as it is used in the code 
- **`classes/`** – Houses Python scripts for different components, such as model training, and visualization.  
- **`pages/`** – Streamlit multi-page support, this conatains the model building and visualization interactive page 
- **`app.py`** – The main entry point for running the Streamlit application, this contains the landing page of the application and has instructions on how to use it.
- **`blr.csv`** – Inference dataset containing yearly composite driving factors for Bangalore 2019.  
- **`delhi.csv`** – Inference dataset containing yearly composite driving factors for Delhi 2019. 
- **`requirements.txt`** – List of dependencies required to run the project. 

## Setup
1. Install streamlit on your system. Some of the dependent libraries that streamlit uses have outdated tools so please check which version works with all the other python libraries it uses. This is mentioned in `requirements.txt`
   ```bash  
   pip install -r requirements.txt  
   ``` 
2. Clone the `webapp` directory locally and in the terminal run `streamlit run app.py`
3. The app should open in your browser and you must be able to navigate it and use it.

## Getting Started
Once the application is up and running, you will see the landing page with more instructions and guidelines on how to use it
![webapp-1](https://github.com/user-attachments/assets/a6fed05e-85f7-416e-910f-b0dcf55cb958)
![webapp-2](https://github.com/user-attachments/assets/8e4fbe3b-504b-4d36-a7bd-bb6abc67e035)
![app-1](https://github.com/user-attachments/assets/cdd16873-55e7-4c82-b251-f66c81d5274e)
![app-2](https://github.com/user-attachments/assets/e749883d-a249-47cb-b02c-ab7b0c8c6317)
![app-3](https://github.com/user-attachments/assets/79c5c2e9-63ca-42d8-a71b-daffde803595)

