#Imports for Geopandas
import geopandas as gpd

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


def extract_damaged_building_coordinates_with_centroids(buildings_geo, collapsed_gpkg):
    """
    Explanation:
    Extract the centroids and damage levels of each damaged building.

    Parameters:
    buildings_geo (geopandas.GeoDataFrame): GeoDataFrame containing building footprints.
    collapsed_gpkg (geopandas.GeoDataFrame): GeoDataFrame containing building information.

    Returns:
    GeoDataFrame: GeoDataFrame containing the centroids and damage levels of damaged (collapsed) buildings.
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

    # Create a new column for centroids
    damaged_building_coordinates['centroid'] = damaged_building_coordinates['geometry'].centroid

    return damaged_building_coordinates


