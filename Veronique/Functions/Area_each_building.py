#Imports for Openstreetmap
import pandas as pd
from shapely.geometry import Polygon

def gen_area_buildings_turkey(area_turkey_OSM, name="some_city"):
    """
    Explanation: 
    Calculate the area for each building footprint in Turkey.

    Parameters: 
    area_turkey_OSM (GeoDataFrame): GeoDataFrame containing building footprints.

    Returns: 
    pd.DataFrame: DataFrame with building IDs and their respective areas in square meters.
    """
    # Correct the orientation of building geometries (make them counterclockwise)
    area_turkey_OSM['geometry'] = area_turkey_OSM['geometry'].apply(lambda geom: Polygon(geom.exterior.coords) if geom.geom_type == 'Polygon' else geom)

    # Extract coordinates (centroid) and area of each building
    building_area_data = []
    for index, geometry in area_turkey_OSM.iterrows():
        if geometry['geometry'].geom_type == 'Polygon':
            centroid = geometry['geometry'].centroid
            area_m2 = geometry['geometry'].area  
            building_area_data.append({
                'Area (m²)': area_m2
            })

    # Create a DataFrame for the building areas
    area_table = pd.DataFrame(building_area_data)
    
    return area_table




