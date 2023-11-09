#Imports for Geopandas
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.wkt import loads

#Imports for statistical analysis
import pandas as pd

#information collapsed buildings
def collapsed_buildings (city_turkey_collapsed):
    """
    Explanation: 
    Print the coordinates of each building and mark the damaged buildings.
        
    Parameters: 
    file_path (str): Path to the GPKG file containing collapsed buildings data
    city_name (str): Name of the city for which to map damaged buildings

    Returns: 
    Datasheet, shape, description
    """
    print((city_turkey_collapsed).head(5))

    # Rows and columns of the islahiye data
    print((city_turkey_collapsed).shape)
    # Descriptive statistics of the islahiye' data
    print((city_turkey_collapsed).describe())

# Print the coordinates of each building, and mark the damaged buildings
# Correct the orientation of building geometries connect the damaged buildings with the geometry of the damaged buildings 
def extract_damaged_building_coordinates(buildings_geo, collapsed_gpkg):
    """
    Explanation:
    Print the coordinates of each damaged building, including their damage levels.

    Parameters:
    collapsed_gpkg (GeoDataFrame): GeoDataFrame containing building information
    buildings_geo (geopandas.GeoDataFrame): GeoDataFrame containing building footprints.

    Returns:
    GeoDataFrame: GeoDataFrame containing the coordinates and damage levels of damaged (collapsed) buildings.
    """
     
    # Reproject buildings_geo to match the CRS of collapsed_gpkg
    buildings_geo_reprojected = buildings_geo.to_crs(collapsed_gpkg.crs)

    # Spatial join to get the coordinates of damaged buildings
    damaged_building_coordinates = gpd.sjoin(buildings_geo_reprojected, collapsed_gpkg, predicate='intersects')

    # Filter out rows where building is 'way'
    damaged_building_coordinates = damaged_building_coordinates[damaged_building_coordinates['building'] == 'yes']

    # Reset index to flatten the DataFrame
    damaged_building_coordinates.reset_index(drop=True, inplace=True)

    # Extract the 'building', 'dmg', and 'geometry' columns
    damaged_building_coordinates = damaged_building_coordinates[['building', 'dmg', 'geometry']]
    
    return damaged_building_coordinates
   
#extract_damaged_building_coordinates(buildings_islahiye, islahiye_collapsed)

# Assuming 'buildings_islahiye' and 'islahiye_collapsed' are GeoDataFrames with the required data
#damaged_building_coordinates_turkey = extract_damaged_building_coordinates(buildings_islahiye, islahiye_collapsed)

# Print the first few rows to verify the data
# print(damaged_building_coordinates_turkey.head())

#print("Coordinates of the first building:")
#print(damaged_building_coordinates_turkey['geometry'][0])

def plot_building_shape(building_df, building_index):
    """
    Explanation:
    Plot the shape of a building based on its index.

    Parameters:
    building_df (geopandas.GeoDataFrame): GeoDataFrame containing building shapes.
    building_index (int): Index of the building to plot.

    Returns:
    None
    """
    # Extract the geometry (polygon) for the specified building
    polygon = building_df['geometry'][building_index]

    # Extract the WKT (Well-Known Text) representation of the polygon
    wkt_polygon = polygon.wkt

    # Parse the WKT representation into x and y arrays
    polygon = loads(wkt_polygon)
    x, y = polygon.exterior.xy

    # Plot the polygon
    plt.figure()
    plt.plot(x, y, color='blue')
    plt.fill(x, y, alpha=0.5, color='lightblue')  # Fill the polygon
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Building Shape (Index: {building_index})')
    plt.grid(True)
    plt.show()

def extract_damaged_building_info_with_centroids(building_df):
    """
    Explanation:
    Extract information for damaged buildings and their centroid coordinates.

    Parameters:
    building_df (geopandas.GeoDataFrame): GeoDataFrame containing building shapes.

    Returns:
    pandas.DataFrame: DataFrame containing building index, building name, and centroid coordinates.
    """
    building_info_list = []

    for idx, row in building_df.iterrows():
        # Use the row index as the building index
        building_index = idx
        building_name = row['building']

        # Extract centroid coordinates
        centroid = row['geometry'].centroid
        centroid_coordinates = (centroid.x, centroid.y)

        # Add the information to the list
        building_info_list.append({
            'Building Index': building_index,
            'Building Name': building_name,
            'Centroid Coordinates': centroid_coordinates
        })

    # Create a DataFrame from the list of dictionaries
    building_info_df = pd.DataFrame(building_info_list)

    return building_info_df

def process_building_data(buildings_gdf, collapsed_gdf, damage_threshold, output_directory):
    """
    Explanation:
    Process building data and save relevant information to CSV files.

    Parameters:
    buildings_gdf (geopandas.GeoDataFrame): GeoDataFrame containing building information.
    collapsed_gdf (geopandas.GeoDataFrame): GeoDataFrame containing collapsed building information.
    damage_threshold (float): Threshold for marking buildings as collapsed based on damage level.
    output_directory (str): Directory where CSV files will be saved.
    """

    # Extract damaged building information with centroids
    damaged_building_info = extract_damaged_building_info_with_centroids(buildings_gdf)

    # Filter buildings based on the damage threshold
    collapsed_building_info = collapsed_gdf[collapsed_gdf['dmg'] >= damage_threshold]

    # Extract damaged building coordinates
    damaged_building_coordinates = extract_damaged_building_coordinates(buildings_gdf, collapsed_building_info)

    # Save DataFrames to CSV files
    damaged_building_info.to_csv(output_directory + 'damaged_building_info.csv', index=False)
    collapsed_building_info.to_csv(output_directory + 'collapsed_building_info.csv', index=False)
    damaged_building_coordinates.to_csv(output_directory + 'damaged_building_coordinates.csv', index=False)







