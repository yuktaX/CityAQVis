import json
import pandas as pd
import folium
from folium.plugins import HeatMap
import plotly.express as px


class Visualiser:
    def __init__(self, model, driving_factors, city) -> None:
        self.model = model
        self.driving_factors = driving_factors
        self.city = city
        
        #input of ground locations points along 
        #with yearly composite of driving factors
        #we will use this as input to predict from our trained model and visualize it
        
        if city == "Bangalore":
            self.grid_df = pd.read_csv("blr.csv")
            self.lat_min, self.lat_max = 12.85, 13.20
            self.lon_min, self.lon_max = 77.45, 77.80
        else:
            self.grid_df = pd.read_csv("del.csv")
            self.lat_min, self.lat_max = 28.40, 28.90
            self.lon_min, self.lon_max = 76.80, 77.30

            
    
    def foliumMap(self):
        features = []
        for key in self.driving_factors:
            if self.driving_factors[key]:
                features.append(key)

        self.grid_df['NO2_prediction'] = self.model.predict(self.grid_df[features])

        # Create a base map centered around the city
        # lat_min, lat_max = 12.85, 13.20
        # lon_min, lon_max = 77.45, 77.80
    
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
            bottom: 50px; left: 50px; width: 150px; height: 150px;
            background-color: white; border:2px solid grey; z-index:9999; font-size:14px;
            padding: 0px;
            ">
            <b>NO2 Levels</b><br>
            <i style="background: rgba(0, 0, 255, 0.5);width: 20px;height: 10px;display: inline-block;"></i> Low (<10 μg/m³)<br>
            <i style="background: rgba(0, 255, 0, 0.5);width: 20px;height: 10px;display: inline-block;"></i> Moderate (10-20 μg/m³)<br>
            <i style="background: rgba(255, 255, 0, 0.5);width: 20px;height: 10px;display: inline-block;"></i> High (20-40 μg/m³)<br>
            <i style="background: rgba(255, 0, 0, 0.5);width: 20px;height: 10px;display: inline-block;"></i> Very High (>40 μg/m³)
        </div>
        '''

        # Add the custom legend to the map
        m.get_root().html.add_child(folium.Element(legend_html))
        
        # Embed NO₂ data in JavaScript format
        no2_data_js = json.dumps(heat_data)
        
        click_js = f"""
        <script>
            document.addEventListener("DOMContentLoaded", function() {{
                var map = this._leaflet_map;
                map.on('click', function(e) {{
                    let lat = e.latlng.lat;
                    let lon = e.latlng.lng;
                    let no2Value = 'No data';
                    let minDist = Infinity;
                    let data = {no2_data_js};

                    data.forEach(point => {{
                        let distance = Math.sqrt((lat - point[0]) ** 2 + (lon - point[1]) ** 2);
                        if (distance < minDist) {{
                            minDist = distance;
                            no2Value = point[2];
                        }}
                    }});

                    // Add marker and popup with NO2 value
                    let marker = L.marker([lat, lon]).addTo(map);
                    marker.bindPopup("NO₂ concentration: " + no2Value.toFixed(3) + " μg/m³").openPopup();
                }});
            }});
        </script>
        """

        # Replace placeholder with the actual NO2 data
        #click_js = click_js.replace("{{ data }}", str(heat_data))

        # Attach the JavaScript function to the map
        m.get_root().html.add_child(folium.Element(click_js))

        map_html = m._repr_html_()
        return map_html
    
    def plotlyMap(self):
        
        features = []
        for key in self.driving_factors:
            if self.driving_factors[key]:
                features.append(key)

        # Prepare data for Plotly
        self.grid_df['NO2_prediction'] = self.model.predict(self.grid_df[features])

        # Define map boundaries
        # lat_min, lat_max = 12.85, 13.20
        # lon_min, lon_max = 77.45, 77.80
        
        heat_data = self.grid_df

        # Create a scatter mapbox plot with a color scale for NO2 predictions
        fig = px.scatter_mapbox(heat_data, lat='latitude', lon='longitude',
                                color='NO2_prediction',
                                color_continuous_scale='Viridis',
                                mapbox_style='open-street-map',
                                size_max=5,  # Smaller point size for denser effect
                                zoom=11,  # Adjust initial zoom level
                            )

        fig.update_layout(
            mapbox=dict(
                center={"lat": (self.lat_min + self.lat_max) / 2, "lon": (self.lon_min + self.lon_max) / 2},
                zoom=9,  # Default zoom
                style="open-street-map",# Limit the display to a fixed geographical range to restrict effective zoom
                layers=[]
            ),
            height=500, 
            width=1000
        )
        # Display the map
        return fig