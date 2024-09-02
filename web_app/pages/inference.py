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

'''
    points = gpd.read_file('../DownloadTest/stations.shp')

    #Driving factors
    points['NO2'] = None
    points['Elevation'] = None
    points['Rainfall'] = None
    points['Population'] = None
    points['VIIRS'] = None
    # points['LandUse'] = None
    points['Temperature'] = None
    points['WindSpeed'] = None

    months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
    no2_raster_files = [f'../DownloadTest/no2_{month}.tif' for month in months]
    rainfall_raster_files = [f'../DownloadTest/rainfall_{month}.tif' for month in months]
    viirs_raster_files = [f'../DownloadTest/viirs_{month}.tif' for month in months]
    wind_raster_files = [f'../DownloadTest/windspeed_{month}.tif' for month in months]
    temp_raster_files = [f'../DownloadTest/temp_{month}.tif' for month in months]

    NO2_raster = rio.open('../DownloadTest/no2Test2.tif')
    NO2_arr = NO2_raster.read(1)
    Elevation_raster = rio.open('../DownloadTest/elevationTest1.tif')
    Elevation_arr = Elevation_raster.read(1)
    Rainfall_raster = rio.open('../DownloadTest/rainTest1.tif')
    Rainfall_arr = Rainfall_raster.read(1)
    Pop_raster = rio.open('../DownloadTest/populationTest1.tif')
    Pop_arr = Pop_raster.read(1)

    # Landuse_raster = rio.open('../DownloadTest/categoricalUse.tif')
    # Landuse_arr = Landuse_raster.read(1)

    count=0

    for index,row in points.iterrows(): #iterate over the points in the shapefile
        longitude=row['geometry'].x #get the longitude of the point
        latitude=row['geometry'].y  #get the latitude of the point

        rowIndex, colIndex = NO2_raster.index(longitude,latitude) # the corresponding pixel to the point (longitude,latitude)

        # Extract the raster values at the point location
        points['NO2'].loc[index] = NO2_arr[rowIndex, colIndex]
        points['Elevation'].loc[index] = Elevation_arr[rowIndex, colIndex]
        points['Population'].loc[index] = Pop_arr[rowIndex, colIndex]
        points['Rainfall'].loc[index] = Rainfall_arr[rowIndex, colIndex]
        # points['LandUse'].loc[index] = Landuse_arr[rowIndex, colIndex]
        #points['VIIRS'].loc[index] = Rainfall_arr[rowIndex, colIndex]

    new_rows = []

    for _, point in points.iterrows():
        for month, no2_file, rainfall_file, viirs_file, temp_file, wind_file in zip(months, no2_raster_files, rainfall_raster_files, viirs_raster_files, temp_raster_files, wind_raster_files):
            no2_value = extract_raster_values(point, no2_file)
            rainfall_value = extract_raster_values(point, rainfall_file)
            viirs_value = extract_raster_values(point, viirs_file)
            temp_value = extract_raster_values(point, temp_file)
            wind_value = extract_raster_values(point, wind_file)
            new_row = {
                'geometry': point['geometry'],
                'NAME': point['NAME'],
                'NO2': no2_value,
                'Elevation': point['Elevation'],
                'Rainfall': rainfall_value,
                'Population': point['Population'],
                'VIIRS':viirs_value,
                # 'LandUse': point['LandUse'],
                'Temperature':temp_value,
                'WindSpeed':wind_value,
            }
            new_rows.append(new_row)

    new_points = gpd.GeoDataFrame(new_rows, crs=points.crs)
'''