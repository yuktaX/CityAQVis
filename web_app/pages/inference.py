import streamlit as st
import ee
import geemap.foliumap as geemap

st.title("Try out your model")
    
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h2 style='text-align: center; color: white;'>City 1</h2>", unsafe_allow_html=True)
    Map1 = geemap.Map(center=(13, 77.5), zoom=9)
    Map1.to_streamlit(height=600)
    
with col2:
    st.markdown("<h2 style='text-align: center; color: white;'>City 2</h2>", unsafe_allow_html=True)
    Map2 = geemap.Map(center=(13, 77.5), zoom=9)
    Map2.to_streamlit(height=600)

# Define the boundaries of the city (latitude and longitude)
lat_min, lat_max = 12.85, 13.20  # Example for a city like Bangalore
lon_min, lon_max = 77.45, 77.80

# Define the grid resolution (distance between points in degrees)
grid_size = 0.01  # Adjust as needed

# Create the grid points
lat_grid = np.arange(lat_min, lat_max, grid_size)
lon_grid = np.arange(lon_min, lon_max, grid_size)
grid_points = [(lat, lon) for lat in lat_grid for lon in lon_grid]

# Convert to DataFrame
grid_df = pd.DataFrame(grid_points, columns=['latitude', 'longitude'])

ee.Authenticate() 
ee.Initialize(project='ee-brijdesai2003')

no2_dataset = ee.ImageCollection('COPERNICUS/S5P/OFFL/L3_NO2') \
              .filterDate('2023-01-01', '2023-01-31') \
              .select('tropospheric_NO2_column_number_density') \
              .mean()  # Get the average for the month

elevation_dataset = ee.Image('CGIAR/SRTM90_V4')

rainfall_dataset = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE") \
                  .filterDate('2023-01-01', '2023-01-31') \
                  .select('pr') \
                  .mean()

population_dataset = ee.ImageCollection("CIESIN/GPWv411/GPW_Population_Count") \
                        .filterDate('2020-01-01', '2020-12-31') \
                        .mean()

viirs_dataset = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMCFG') \
                        .filterDate('2023-01-01', '2023-01-31') \
                        .select('avg_rad') \
                        .mean()

# landuse_dataset = ee.Image("JRC/GHSL/P2023A/GHS_BUILT_C/2018")

temperature_dataset = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE") \
                  .filterDate('2023-01-01', '2023-01-31') \
                  .select('tmmx') \
                  .mean()

windspeed_dataset = ee.ImageCollection("IDAHO_EPSCOR/TERRACLIMATE") \
                  .filterDate('2023-01-01', '2023-01-31') \
                  .select('vs') \
                  .mean()

# Define a function to get data from Earth Engine for a single point
def get_ee_data(lat, lon):
    point = ee.Geometry.Point(lon, lat)

    # Get NO2 data from TROPOMI
    no2_value = no2_dataset.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=3000
    ).get('tropospheric_NO2_column_number_density').getInfo()

    # Example: Get Elevation data
    elevation_value = elevation_dataset.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=3000
    ).get('elevation').getInfo()

    # Get Rainfall data from TerraClimate
    rainfall_value = rainfall_dataset.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=3000
    ).get('pr').getInfo()

    # Get Population data from CIESIN

    population_value = population_dataset.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=3000
    ).get('population_count').getInfo()

    # Get VIIRS Nighttime Lights data
    viirs_value = viirs_dataset.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=3000
    ).get('avg_rad').getInfo()

     # Get Land Use data from GHS Built-Up Grid
    # landuse_value = landuse_dataset.reduceRegion(
    #     reducer=ee.Reducer.mean(),
    #     geometry=point,
    #     scale=1000
    # ).get('built_characteristics').getInfo()

    temperature_value = temperature_dataset.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=3000
    ).get('tmmx').getInfo()

    windspeed_value = windspeed_dataset.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point,
        scale=3000
    ).get('vs').getInfo()

    return no2_value, elevation_value, rainfall_value, population_value, viirs_value, temperature_value, windspeed_value

    # Add other data sources similarly...

grid_df[['NO2 (mol/m^2)', 'Elevation', 'Rainfall', 'Population', 'VIIRS', 'Temperature', 'WindSpeed']] = grid_df.apply(lambda row: get_ee_data(row['latitude'], row['longitude']), axis=1, result_type='expand')

csv_filename = 'blr.csv'
grid_df.to_csv(csv_filename, index=False)

features = ['NO2 (mol/m^2)', 'Elevation', 'Rainfall', 'Population', 'VIIRS', 'Temperature', 'WindSpeed']
grid_df['NO2_prediction'] = best_model.predict(grid_df[features])

import folium
from folium.plugins import HeatMap

# Create a base map centered around the city
m = folium.Map(location=[(lat_min + lat_max) / 2, (lon_min + lon_max) / 2], zoom_start=12)

# Convert predictions to a list of [latitude, longitude, NO2] for HeatMap
heat_data = [[row['latitude'], row['longitude'], row['NO2_prediction']] for index, row in grid_df.iterrows()]

# Add the heatmap layer
HeatMap(heat_data, radius=10).add_to(m)

# Save map to HTML file or display directly
m.save('no2_heatmap.html')
m
