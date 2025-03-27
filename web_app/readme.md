# CityAQVis - Air Quality Visualization Platform  

This repository contains the code and data for CityAQVis, a machine learning-based platform for air quality prediction and visualization.  

## Project Structure  

- **`.streamlit/`** – Streamlit configuration files for customizing the web application settings.  
- **`.venv/`** – Virtual environment directory containing dependencies (not required for deployment).  
- **`Data/`** – Contains data files required for model training and visualization.  
- **`classes/`** – Houses Python scripts for different components, such as model training, and visualization.  
- **`pages/`** – Streamlit multi-page support, containing different views of the application.  
- **`app.py`** – The main entry point for running the Streamlit application, this contains the landing page of the application and has instructions on how to use it.
- **`blr.csv`** – Dataset containing air quality data for Bangalore 2019.  
- **`delhi.csv`** – Dataset containing air quality data for Delhi 2019.  
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

