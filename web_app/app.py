import streamlit as st
import ee
import geemap.foliumap as geemap

st.set_page_config(layout="wide")
st.title("CityAQVis: ML Sandbox With Comparative Visual Analytics For Air Quality In Urban Regions")

# Sidebar navigation with HTML links
st.sidebar.title("Navigation")
st.sidebar.markdown("[Key Features](#key-features)")
st.sidebar.markdown("[How to Use](#how-to-use)")
st.sidebar.markdown("[Technology Stack](#technology-stack)")
st.sidebar.markdown("[Extend for other Cities](#extension-cities)")
st.sidebar.markdown("[Future Enhancements](#future-enhancements)")

st.markdown(""" --- """)

# Introduction
st.markdown(
    """
    <span style="font-size:20px;">
 
    Welcome to the Air Quality Prediction and Comparision Web App 🌍!  
    This application allows you to predict air quality metrics based on user-selected cities, pollutants, and driving factors. 
    With machine learning models, you can analyze, visualize, and understand pollutant behavior and its influencing factors.
    </span>
    """,
    unsafe_allow_html=True
)

st.markdown(""" --- """)

# Key Features Section
st.markdown("<a id='key-features'></a>", unsafe_allow_html=True)
st.header("📊 Key Features")
st.markdown(
    """
    1. **City-Specific Analysis**  
       - Choose between **Bangalore** and **Delhi** for city-specific air quality predictions.  
       - Tailored data options for different years of study.  

    2. **Pollutant Selection**  
       - Predict metrics for key pollutants, including:
         - **NO2** (Nitrogen Dioxide)
         - **SO2** (Sulfur Dioxide)
         - **CH4** (Methane)

    3. **Driving Factors**  
       - Customize the analysis by selecting driving factors, such as:
         - Elevation
         - Rainfall
         - Population density
         - Night-time lights (VIIRS)
         - Temperature
         - Wind speed

    4. **Model Selection**  
       - Experiment with various machine learning models:
         - **Linear Regression**
         - **Support Vector Machine (SVM)**
         - **Random Forest**
         - **Gradient Boosting Regressor (GBR)**

    5. **Interactive Results and Visualization**  
       - View metrics like **R2 score**, **MAE**, **MSE**, **MAPE**, and **RMSE** after model training.  
       - Explore interactive maps and visualizations using:
         - **Folium**: Geospatial map integration.
         - **Plotly**: Dynamic charts for deeper insights.
    ---
    """
)

# How to Use Section
st.markdown("<a id='how-to-use'></a>", unsafe_allow_html=True)
st.header("🚀 How to Use the App")
st.markdown(
    """
    1. **Select City, Model, and Pollutant**  
       - Choose the city, pollutant, and machine learning model for prediction.

    2. **Choose Year of Study**  
       - Select the year of data analysis, tailored for the chosen city.

    3. **Pick Driving Factors**  
       - Use the multiselect dropdown to select factors affecting air quality.

    4. **Train the Model**  
       - Click the **Train model** button to train the selected model using the provided data.

    5. **Review Results**  
       - Examine the model's performance metrics and explore visualizations of the predictions.
    ---
    """
)

# Technology Stack Section
st.markdown("<a id='technology-stack'></a>", unsafe_allow_html=True)
st.header("🛠️ Technology Stack")
st.markdown(
    """
    - **Frontend**: Built with **Streamlit** for a seamless interactive experience.  
    - **Backend**: Machine learning models implemented in Python using sklearn library.
    - **Visualization**: Folium and Plotly for rich geospatial and graphical outputs.
    ---
    """
)

# How to use for other Cities
st.markdown("<a id='extension-cities'></a>", unsafe_allow_html=True)
st.header("Extension for other cities")
st.markdown(
    """
    1. **Download Station Data**  
       - Download all the station data from the CCRAQM portal or any other source of ground data that you have.
    
    2. **Create CSV with Station Details**  
       - Create a CSV file that includes the name of the stations/ground locations, UTMX, and UTMY.

    3. **Convert CSV to Shapefile**  
       - Convert this CSV into a shapefile using any software like **QGIS**.

    4. **Extract Driving Factor Data**  
       - Extract the driving factor data (in TIFF format) for the stations using the shapefile in **Google Earth Engine**.

    5. **Dataset Preparation in Notebook**  
       - Open the **CityAQVis_Dataset_ML_Model.ipynb** notebook.  
       - Load all the data in the appropriate paths and run the first section: *Dataset Preparation*.

    6. **Download New Points CSV**  
       - Download the `new_points.csv` file. This contains the monthly driving factor data for each station.

    7. **Manually Add Ground Data**  
       - Add the monthly averages of station NO2 data/ground data to the corresponding rows of `new_points.csv`.

    8. **Upload Data and Train Model**  
       - Upload the updated file to the appropriate location or path.  
       - Continue to run the notebook to train any model of your choice.

    9. **Integrate Data with Web App**  
       - If you want to directly upload data to the web app:  
         - Place the data in the `web_app/Data` folder.  
         - Update the paths accordingly in `web_app/classes/ModelTrainer.py`.

    ---
    """
)

# Future Enhancements Section
st.markdown("<a id='future-enhancements'></a>", unsafe_allow_html=True)
st.header("🌟 Future Enhancements")
st.markdown(
    """
    - Add more cities and pollutants for broader analysis.  
    - Introduce additional machine learning models and hyperparameter tuning options.  
    - Expand visualization capabilities with time-series analysis.
    ---
    """
)

# Closing Note
st.markdown(
    """
    Dive in and explore the factors influencing air quality with our powerful prediction tool! 🌱
    """
)
