#Imports for Openstreetmap
import pandas as pd

# Extract coordinates (centroid) of each building
def gen_building_coordinates_turkey(coordinates_turkey):
    """
    Explanation: 
    Extract coordinates (centroid) of each building
        
    Parameters: 
    buildings_Turkey (geopandas.GeoDataFrame): GeoDataFrame containing building footprints.

    Returns: 
    List of tuples: Coordinates (centroid) of each building (latitude, longitude)
    """
    latitudes = []
    longitudes = []

    for index, geometry in coordinates_turkey.iterrows():
        centroid = geometry['geometry'].centroid
        latitudes.append(centroid.y)
        longitudes.append(centroid.x)

    # Print the coordinates of each building
    i_list = []
    for i, (lat, lon) in enumerate(zip(latitudes, longitudes), start=1):
        i_list.append(i)
        #print(f"Building {i} Coordinate: Latitude: {lat}, Longitude: {lon}")
    
    coordinates_df = pd.DataFrame({"Building": i_list,'Longitude': longitudes, 'Latitude': latitudes})

    #print(coordinates_df)
    return coordinates_df


    





