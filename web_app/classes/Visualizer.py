import os
import pandas as pd
import folium
from folium.plugins import HeatMap
import plotly.express as px
import plotly.graph_objects as go

class Visualiser:
    def __init__(self, model, driving_factors, city, year) -> None:
        self.model = model
        self.driving_factors = driving_factors
        self.city = city
        
        # Input of ground locations points along 
        # with yearly composite of driving factors
        # We will use this as input to predict from our trained model and visualize it
        
        # Define root of the repo (1 level above web_app/)
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Set file path based on city and year
        if city == "Bangalore":
            if year == "2019":
                filename = "blr_2019_inference.csv"
            else:
                filename = "blr_2022_inference.csv"
            self.lat_min, self.lat_max = 12.85, 13.20
            self.lon_min, self.lon_max = 77.45, 77.80
        else:  # Delhi
            filename = "delhi_2019_inference.csv"
            self.lat_min, self.lat_max = 28.40, 28.90
            self.lon_min, self.lon_max = 76.80, 77.30

        # Join the full path
        file_path = os.path.join(project_root, "Data", filename)

        # Read the CSV
        self.grid_df = pd.read_csv(file_path)
            
    
    def foliumMap(self):
        features = []
        for key in self.driving_factors:
            if self.driving_factors[key]:
                features.append(key)

        self.grid_df['NO2_prediction'] = self.model.predict(self.grid_df[features])
    
        min_zoom, max_zoom = 1, 13

        # Create a base map centered around the city
        m = folium.Map(location=[(self.lat_min + self.lat_max) / 2, (self.lon_min + self.lon_max) / 2], zoom_start=12, min_zoom=min_zoom, max_zoom=max_zoom)

        # Convert predictions to a list of [latitude, longitude, NO2] for HeatMap
        heat_data = [[row['latitude'], row['longitude'], row['NO2_prediction']] for index, row in self.grid_df.iterrows()]

        # Add the heatmap layer with NO2 predictions
        HeatMap(heat_data, radius=20, blur=25, max_zoom=12, min_opacity=0.4).add_to(m)

        # HTML for the custom legend
        legend_html = '''
        <div style="
            position: fixed;
            bottom: 20px; left: 20px; width: 140px; height: 110px;
            background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
            padding: 0px;
            ">
            <b>NO₂ (µg/m³)</b><br>
            <i style="background: rgba(0, 0, 255, 0.5);width: 20px;height: 10px;display: inline-block;"></i> Low (<10)<br>
            <i style="background: rgba(0, 255, 0, 0.5);width: 20px;height: 10px;display: inline-block;"></i> Moderate (10-20)<br>
            <i style="background: rgba(255, 255, 0, 0.5);width: 20px;height: 10px;display: inline-block;"></i> High (20-40)<br>
            <i style="background: rgba(255, 0, 0, 0.5);width: 20px;height: 10px;display: inline-block;"></i> Very High (>40)
        </div>
        '''

        # Add the custom legend to the map
        m.get_root().html.add_child(folium.Element(legend_html))
        map_html = m._repr_html_()
        return map_html

    def plotlyMap(self, global_scale = False):
        features = [key for key in self.driving_factors if self.driving_factors[key]]

        # Prepare data for Plotly
        self.grid_df['NO2_prediction'] = self.model.predict(self.grid_df[features])

        heat_data = self.grid_df

        # Define global color scale range (fixed for consistency)
        color_min, color_max = (0, 50) if global_scale else (heat_data["NO2_prediction"].min(), heat_data["NO2_prediction"].max())

        # Create a scatter mapbox plot with a fixed color scale for NO2 predictions
        fig = px.scatter_mapbox(
            heat_data, lat='latitude', lon='longitude',
            color='NO2_prediction',
            color_continuous_scale='Viridis',
            range_color=[color_min, color_max],  # FIXED color range for consistency
            mapbox_style='open-street-map',
            size_max=5,
            zoom=11,
        )

        fig.update_layout(
            mapbox=dict(
                center={"lat": (self.lat_min + self.lat_max) / 2, "lon": (self.lon_min + self.lon_max) / 2},
                zoom=9,
                style="open-street-map",
                layers=[]
            ),
            height=600,
            width=1000,
            coloraxis_colorbar=dict(
                title="NO₂ (µg/m³)",
                tickvals=[0, 10, 20, 40, 50] if global_scale else None,  # Ensure tick values align with legend
                ticktext=["Low (<10)", "Moderate (10-20)", "High (20-40)", "Very High (>40)"] if global_scale else None,
            ),
        )
        
        fig.update_layout()

        return fig

