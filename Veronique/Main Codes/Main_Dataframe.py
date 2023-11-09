# Imports for Openstreetmap
import osmnx as ox
# Imports for Geopandas
import geopandas as gpd
# Imports for statistical analysis
import geopandas as gpd
import pandas as pd
import numpy as np
from pyproj import Proj

# Imports for functions
# 1 Geographical location
from Coordinates_footprints_turkey import gen_building_coordinates_turkey
# 2 Damage classification
# 2.1 Location damaged buildings
from Coordinates_damaged_buildings import extract_damaged_building_coordinates, extract_damaged_building_coordinates_with_centroids
# 3 Area
from Area_each_building import gen_area_buildings_turkey
# 4 Building properties
from Properties_Building_Dataframe import gen_buildingdata_gaziantep, gen_buildingdata_kahramanmaraş

# Define tags for buildings
tags = {'building': True}

# Replace this for a path where you want to save the outcomes of this code on your own computer
Own_filepath = r"C:\Users\Véronique\OneDrive\Documenten\Master bouwkunde\Leerjaar 4\CORE"
# Create a Back-End folder with the following folders in there:
# Area excel files
# Building properties
    # Islahiye
    # Nurdagi
    # Marash
    # Turkoglu
# Created Maps
    # Damage Maps
        # Collapse cluster
        # Damage
        # Damage centre
        # Damage orientation
        # Damage visualisation
    # Open Street Maps
# functions
# txt_files
    # Islahiye
    # Marash
    # Nurdagi
    # Turkoglu

# 1. Geographical location of each building
# Gaziantep, Islahiye
buildings_islahiye = ox.features.features_from_address('Şht. Zafer Yılmaz Cd. 15, 27800 İslahiye', tags, dist=10418) 
# Gaziantep, Nurdagi
buildings_nurdagi = ox.features.features_from_address('Cengiz Topel Cd. 1, 27840 Nurdağı/Gaziantep', tags, dist=4100) 
# Kahramanmaraş, Marash
# Manually obtain the coordinates (latitude and longitude) for the address
latitude_marash = 37.574038
longitude_marash = 36.920378 
# Use the coordinates to retrieve OSM data
buildings_marash = ox.features.features_from_point((latitude_marash, longitude_marash), tags, dist=3027)
# Kahramanmaraş, Turkoglu
# Manually obtain the coordinates (latitude and longitude) for the address
latitude_turkoglu = 37.384735
longitude_turkoglu = 36.847980
# Use the coordinates to retrieve OSM data
buildings_turkoglu = ox.features.features_from_point((latitude_turkoglu, longitude_turkoglu), tags, dist=3610)

# 2. Damage Data
# Read Islahiye colapsed buildings
file_islahiye = Own_filepath + r"\Datasets cities in turkey\islahiye_osm_damage_2_7_2023.gpkg"
islahiye_collapsed = gpd.read_file(file_islahiye)
# Read Nurdagi colapsed buildings
file_nurdagi = Own_filepath + r"\Datasets cities in turkey\nurdagi_msft_damage_2_7_2023.gpkg"
nurdagi_collapsed = gpd.read_file(file_nurdagi)
# Read Marash colapsed buildings
file_marash = Own_filepath + r"\Datasets cities in turkey\new_marash_collapsed.gpkg"
marash_collapsed = gpd.read_file(file_marash)
# Read Turkoglu colapsed buildings 
file_turkoglu =  Own_filepath + r"\Datasets cities in turkey\new_turkoglu_collapsed.gpkg"
turkoglu_collapsed = gpd.read_file(file_turkoglu)

# 3. Check the coordinate systems
# Check the CRS of buildings_geo
print("CRS of buildings_geo:", buildings_islahiye.crs)
# Check the CRS of collapsed_gpkg
print("CRS of collapsed_gpkg:", islahiye_collapsed.crs)
# If the CRS is not set or incorrect, you can define it as follows:
# For example, if the CRS is EPSG:4326 (WGS 84), you can define it like this:
buildings_islahiye.crs = 'EPSG:4326'
islahiye_collapsed.crs = 'EPSG:4326'

# 4. Creating new columns for the dataframe
# Creating a new column for in the building properties dataframe
# Hazardous materials is based on construction type and construction year
def has_hazardous_materials(row):
    if row['Construction Type'] == 'Reinforced Concrete' and (row['Construction Year'] == 'Unknown' or row['Construction Year'] <= 1993):
        return True
    else:
        return False

# Creating a new column for in the building properties dataframe
# Occupants is based on area and number of stories
def calculate_occupants(row):
    # Replace this formula with the appropriate calculation based on your needs
    # For example, you can use a formula that considers area and levels
    occupants = round(row['Area (m²)'] * row['Number of Stories'] / 100, 0)  # Adjust the formula as needed
    return occupants

# 5. Creating Dataframes
# 5.1.0 Creating a dataframe for Islahiye with the geographical coordinations, damage levels and building characteristics
building_data_islahiye = gen_buildingdata_gaziantep(count=13213)
df_islahiye = pd.DataFrame(building_data_islahiye)

# 5.1.1 Adding new columns to the Islahiye dataframe
df_islahiye['Hazardous materials'] = df_islahiye.apply(has_hazardous_materials, axis=1)
# Coordinates dataframe Islahiye
coordinates_df_islahiye = gen_building_coordinates_turkey(buildings_islahiye)
# 5.1.2 Area
result_table_area_islahiye = gen_area_buildings_turkey(buildings_islahiye)

# 5.1.3 Combining the dataframes coordinates from OSMX and Building properties
islahiye_info_df = pd.concat([df_islahiye, coordinates_df_islahiye], axis=1)
islahiye_info_df.Latitude = np.around(islahiye_info_df.Latitude, 9)
islahiye_info_df.Longitude = np.around(islahiye_info_df.Longitude, 9)

# 5.1.4 Area
islahiye_info_df['Area (m²)'] = result_table_area_islahiye['Area (m²)']
islahiye_info_df['Area (m²)'] = islahiye_info_df['Area (m²)'] * 10000000000
# 5.1.5 Occupants
islahiye_info_df['Occupants'] = islahiye_info_df.apply(calculate_occupants, axis=1)

# 5.1.6 Modifying the dataframe
islahiye_info_df = islahiye_info_df.drop('Building', axis=1)

# 5.1.7 Damage classification
i_list_islahiye_damage = list(range(1, len(islahiye_collapsed) + 1))
coordinates_islahiye_damage_df = pd.DataFrame({"Building Name": i_list_islahiye_damage, "Damage Level": islahiye_collapsed['dmg'], 'Geometry': islahiye_collapsed['geometry']})

# 1. Geographical location of each building
# Gaziantep, Islahiye
buildings_islahiye = ox.features.features_from_address('Şht. Zafer Yılmaz Cd. 15, 27800 İslahiye', tags, dist=10418) 
# 2. Damage Data
# Read Islahiye colapsed buildings
file_islahiye = Own_filepath + r"\Datasets cities in turkey\islahiye_osm_damage_2_7_2023.gpkg"
islahiye_collapsed = gpd.read_file(file_islahiye)

# Assuming 'buildings_islahiye' and 'islahiye_collapsed' are GeoDataFrames with the required data
damaged_building_coordinates_turkey = extract_damaged_building_coordinates(buildings_islahiye, islahiye_collapsed)
damaged_building_coordinates_centroids = extract_damaged_building_coordinates_with_centroids(buildings_islahiye, islahiye_collapsed)

# From UTM to long/latitude -> Change the coordinate system
myProj = Proj("+proj=utm +zone=37 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
# The next cell uses the object created (myProj) to convert the UTM positions to longitude and latitude. That is why we need the inverse keyword.
lon, lat = myProj(damaged_building_coordinates_centroids.centroid.apply(lambda p: p.x).values, damaged_building_coordinates_centroids.centroid.apply(lambda p: p.y).values, inverse=True)

damage_islahiye_df = pd.DataFrame({'Building Name': damaged_building_coordinates_centroids['building'].tolist(), 'Damage Level': damaged_building_coordinates_centroids['dmg'], 'Longitude': np.around(lat,9), 'Latitude': np.around(lon,9)})

# Check overlap between the two tables, overlap is based on the coordinates
overlap_damage_coordinates_islahiye = islahiye_info_df.merge(damage_islahiye_df, on=['Longitude', 'Latitude'], how='outer')
overlap_damage_coordinates_islahiye = overlap_damage_coordinates_islahiye[overlap_damage_coordinates_islahiye['Construction Type'].notna()]

# Since the damage dataset does not show the undamaged data, there was a difference in the amount of buildings
overlap_damage_coordinates_islahiye['Damage Level'] = overlap_damage_coordinates_islahiye['Damage Level'].fillna(0)
overlap_damage_coordinates_islahiye.drop_duplicates(subset=['Longitude', 'Latitude'], keep='first', inplace=True)

overlap_damage_coordinates_islahiye.rename(columns={'Building Name_x': 'Building Name'}, inplace=True)
overlap_damage_coordinates_islahiye = overlap_damage_coordinates_islahiye.drop('Building Name_y', axis=1)

# 5.1.8. Save the output
Total_df_islahiye_info = overlap_damage_coordinates_islahiye[['Building Name', 'Damage Level', 'Longitude', 'Latitude', 'Area (m²)', 'Number of Stories', 'Occupants', 'Construction Year', 'Construction Type', 'Hazardous materials']]
Total_df_islahiye_info.to_excel(r"C:\Users\Véronique\OneDrive\Documenten\Master bouwkunde\Leerjaar 4\CORE\09_Back-End\Dataframes\Islahiye\Df_Islahiye.xlsx", index=False)

gpf_islahiye = gpd.GeoDataFrame(Total_df_islahiye_info, geometry=gpd.points_from_xy(Total_df_islahiye_info['Longitude'], Total_df_islahiye_info['Latitude']))
geojson_filepath = r"C:\Users\Véronique\OneDrive\Documenten\Master bouwkunde\Leerjaar 4\CORE\09_Back-End\Dataframes\Islahiye\Df_Islahiye.geojson"
gpf_islahiye.to_file(geojson_filepath, driver='GeoJSON')

# 5.2.0 Creating a dataframe for Nurdagi with the geographical coordinations, damage levels and building characteristics
building_data_nurdagi = gen_buildingdata_gaziantep(count=4537)
df_nurdagi = pd.DataFrame(building_data_nurdagi)

# 5.2.1 Adding new columns to the nurdagi dataframe
df_nurdagi['Hazardous materials'] = df_nurdagi.apply(has_hazardous_materials, axis=1)
# Coordinates dataframe nurdagi
coordinates_df_nurdagi = gen_building_coordinates_turkey(buildings_nurdagi)
# 5.2.2 Area
result_table_area_nurdagi = gen_area_buildings_turkey(buildings_nurdagi)

# 5.2.3 Combining the dataframes coordinates from OSMX and Building properties
nurdagi_info_df = pd.concat([df_nurdagi, coordinates_df_nurdagi], axis=1)
nurdagi_info_df.Latitude = np.around(nurdagi_info_df.Latitude, 9)
nurdagi_info_df.Longitude = np.around(nurdagi_info_df.Longitude, 9)

# 5.2.4 Area
nurdagi_info_df['Area (m²)'] = result_table_area_nurdagi['Area (m²)']
nurdagi_info_df['Area (m²)'] = nurdagi_info_df['Area (m²)'] * 10000000000
# 5.2.5 Occupants
nurdagi_info_df['Occupants'] = nurdagi_info_df.apply(calculate_occupants, axis=1)

# 5.2.6 Modifying the dataframe
nurdagi_info_df = nurdagi_info_df.drop('Building', axis=1)

# 5.2.7 Damage classification
i_list_nurdagi_damage = list(range(1, len(nurdagi_collapsed) + 1))
coordinates_nurdagi_damage_df = pd.DataFrame({"Building Name": i_list_nurdagi_damage, "Damage Level": nurdagi_collapsed['dmg'], 'Geometry': nurdagi_collapsed['geometry']})

# 1. Geographical location of each building
# Gaziantep, Nurdagi
buildings_nurdagi = ox.features.features_from_address('Cengiz Topel Cd. 1, 27840 Nurdağı/Gaziantep', tags, dist=4100) 

# 2. Damage Data
# Read Nurdagi colapsed buildings
file_nurdagi = Own_filepath + r"\Datasets cities in turkey\nurdagi_msft_damage_2_7_2023.gpkg"
nurdagi_collapsed = gpd.read_file(file_nurdagi)

# Assuming 'buildings_nurdagi' and 'nurdagi_collapsed' are GeoDataFrames with the required data
damaged_building_coordinates_turkey = extract_damaged_building_coordinates(buildings_nurdagi, nurdagi_collapsed)
damaged_building_coordinates_centroids = extract_damaged_building_coordinates_with_centroids(buildings_nurdagi, nurdagi_collapsed)

# From UTM to long/latitude -> Change the coordinate system
myProj = Proj("+proj=utm +zone=37 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
# The next cell uses the object created (myProj) to convert the UTM positions to longitude and latitude. That is why we need the inverse keyword.
lon, lat = myProj(damaged_building_coordinates_centroids.centroid.apply(lambda p: p.x).values, damaged_building_coordinates_centroids.centroid.apply(lambda p: p.y).values, inverse=True)

damage_nurdagi_df = pd.DataFrame({'Building Name': damaged_building_coordinates_centroids['building'].tolist(), 'Damage Level': damaged_building_coordinates_centroids['dmg'], 'Longitude': np.around(lat,9), 'Latitude': np.around(lon,9)})

# Check overlap between the two tables, overlap is based on the coordinates
overlap_damage_coordinates_nurdagi = nurdagi_info_df.merge(damage_nurdagi_df, on=['Longitude', 'Latitude'], how='outer')
overlap_damage_coordinates_nurdagi = overlap_damage_coordinates_nurdagi[overlap_damage_coordinates_nurdagi['Construction Type'].notna()]

# Since the damage dataset does not show the undamaged data, there was a difference in the amount of buildings
overlap_damage_coordinates_nurdagi['Damage Level'] = overlap_damage_coordinates_nurdagi['Damage Level'].fillna(0)
overlap_damage_coordinates_nurdagi.drop_duplicates(subset=['Longitude', 'Latitude'], keep='first', inplace=True)

overlap_damage_coordinates_nurdagi.rename(columns={'Building Name_x': 'Building Name'}, inplace=True)
overlap_damage_coordinates_nurdagi = overlap_damage_coordinates_nurdagi.drop('Building Name_y', axis=1)

# 5.2.8. Save the output
Total_df_nurdagi_info = overlap_damage_coordinates_nurdagi[['Building Name', 'Damage Level', 'Longitude', 'Latitude', 'Area (m²)', 'Number of Stories', 'Occupants', 'Construction Year', 'Construction Type', 'Hazardous materials']]
Total_df_nurdagi_info.to_excel(r"C:\Users\Véronique\OneDrive\Documenten\Master bouwkunde\Leerjaar 4\CORE\09_Back-End\Dataframes\Nurdagi\Df_Nurdagi.xlsx", index=False)

gpf_nurdagi = gpd.GeoDataFrame(Total_df_nurdagi_info, geometry=gpd.points_from_xy(Total_df_nurdagi_info['Longitude'], Total_df_nurdagi_info['Latitude']))
geojson_filepath = r"C:\Users\Véronique\OneDrive\Documenten\Master bouwkunde\Leerjaar 4\CORE\09_Back-End\Dataframes\Nurdagi\Df_Nurdagi.geojson"
gpf_nurdagi.to_file(geojson_filepath, driver='GeoJSON')

# 5.3.0 Creating a dataframe for Marash with the geographical coordinations, damage levels and building characteristics
building_data_marash = gen_buildingdata_kahramanmaraş(count=40375)
df_marash = pd.DataFrame(building_data_marash)

# 5.3.1 Adding new columns to the marash dataframe
df_marash['Hazardous materials'] = df_marash.apply(has_hazardous_materials, axis=1)
# Coordinates dataframe marash
coordinates_df_marash = gen_building_coordinates_turkey(buildings_marash)

# 5.3.2 Area
result_table_area_marash = gen_area_buildings_turkey(buildings_marash)

# 5.3.3 Combining the dataframes coordinates from OSMX and Building properties
marash_info_df = pd.concat([df_marash, coordinates_df_marash], axis=1)
marash_info_df.Latitude = np.around(marash_info_df.Latitude, 9)
marash_info_df.Longitude = np.around(marash_info_df.Longitude, 9)

# 5.3.4 Area
marash_info_df['Area (m²)'] = result_table_area_marash['Area (m²)']
marash_info_df['Area (m²)'] = marash_info_df['Area (m²)'] * 10000000000

# 5.3.5 Occupants
marash_info_df['Occupants'] = marash_info_df.apply(calculate_occupants, axis=1)

# 5.3.6 Modifying the dataframe
marash_info_df = marash_info_df.drop('Building', axis=1)

# 5.3.7 Damage classification
i_list_marash_damage = list(range(1, len(marash_collapsed) + 1))
coordinates_marash_damage_df = pd.DataFrame({"Building Name": i_list_marash_damage, "Damage Level": marash_collapsed['dmg'], 'Geometry': marash_collapsed['geometry']})

# 1. Geographical location of each building
# Kahramanmaraş, Marash
latitude_marash = 37.574038
longitude_marash = 36.920378 
# Use the coordinates to retrieve OSM data
buildings_marash = ox.features.features_from_point((latitude_marash, longitude_marash), tags, dist=3027)

# 2. Damage Data
# Read Marash colapsed buildings
file_marash = Own_filepath + r"\Datasets cities in turkey\new_marash_collapsed.gpkg"
marash_collapsed = gpd.read_file(file_marash)

# Assuming 'buildings_marash' and 'marash_collapsed' are GeoDataFrames with the required data
damaged_building_coordinates_turkey = extract_damaged_building_coordinates(buildings_marash, marash_collapsed)
damaged_building_coordinates_centroids = extract_damaged_building_coordinates_with_centroids(buildings_marash, marash_collapsed)

# From UTM to long/latitude -> Change the coordinate system
myProj = Proj("+proj=utm +zone=37 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
# The next cell uses the object created (myProj) to convert the UTM positions to longitude and latitude. That is why we need the inverse keyword.
lon, lat = myProj(damaged_building_coordinates_centroids.centroid.apply(lambda p: p.x).values, damaged_building_coordinates_centroids.centroid.apply(lambda p: p.y).values, inverse=True)

damage_marash_df = pd.DataFrame({'Building Name': damaged_building_coordinates_centroids['building'].tolist(), 'Damage Level': damaged_building_coordinates_centroids['dmg'], 'Longitude': np.around(lat,9), 'Latitude': np.around(lon,9)})

# Check overlap between the two tables, overlap is based on the coordinates
overlap_damage_coordinates_marash = marash_info_df.merge(damage_marash_df, on=['Longitude', 'Latitude'], how='outer')
overlap_damage_coordinates_marash = overlap_damage_coordinates_marash[overlap_damage_coordinates_marash['Construction Type'].notna()]

# Since the damage dataset does not show the undamaged data, there was a difference in the amount of buildings
overlap_damage_coordinates_marash['Damage Level'] = overlap_damage_coordinates_marash['Damage Level'].fillna(0)
overlap_damage_coordinates_marash.drop_duplicates(subset=['Longitude', 'Latitude'], keep='first', inplace=True)

overlap_damage_coordinates_marash.rename(columns={'Building Name_x': 'Building Name'}, inplace=True)
overlap_damage_coordinates_marash = overlap_damage_coordinates_marash.drop('Building Name_y', axis=1)

# 5.3.8. Save the output
Total_df_marash_info = overlap_damage_coordinates_marash[['Building Name', 'Damage Level', 'Longitude', 'Latitude', 'Area (m²)', 'Number of Stories', 'Occupants', 'Construction Year', 'Construction Type', 'Hazardous materials']]
Total_df_marash_info.to_excel(r"C:\Users\Véronique\OneDrive\Documenten\Master bouwkunde\Leerjaar 4\CORE\09_Back-End\Dataframes\Marash\Df_Marash.xlsx", index=False)

gpf_marash = gpd.GeoDataFrame(Total_df_marash_info, geometry=gpd.points_from_xy(Total_df_marash_info['Longitude'], Total_df_marash_info['Latitude']))
geojson_filepath = r"C:\Users\Véronique\OneDrive\Documenten\Master bouwkunde\Leerjaar 4\CORE\09_Back-End\Dataframes\Marash\Df_Marash.geojson"
gpf_marash.to_file(geojson_filepath, driver='GeoJSON')

# 5.4.0 Creating a dataframe for Turkoglu with the geographical coordinations, damage levels and building characteristics
building_data_turkoglu = gen_buildingdata_kahramanmaraş(count=3816)
df_turkoglu = pd.DataFrame(building_data_turkoglu)

# 5.4.1 Adding new columns to the turkoglu dataframe
df_turkoglu['Hazardous materials'] = df_turkoglu.apply(has_hazardous_materials, axis=1)
# Coordinates dataframe turkoglu
coordinates_df_turkoglu = gen_building_coordinates_turkey(buildings_turkoglu)

# 5.4.2 Area
result_table_area_turkoglu = gen_area_buildings_turkey(buildings_turkoglu)

# 5.4.3 Combining the dataframes coordinates from OSMX and Building properties
turkoglu_info_df = pd.concat([df_turkoglu, coordinates_df_turkoglu], axis=1)
turkoglu_info_df.Latitude = np.around(turkoglu_info_df.Latitude, 9)
turkoglu_info_df.Longitude = np.around(turkoglu_info_df.Longitude, 9)

# 5.4.4 Area
turkoglu_info_df['Area (m²)'] = result_table_area_turkoglu['Area (m²)']
turkoglu_info_df['Area (m²)'] = turkoglu_info_df['Area (m²)'] * 10000000000

# 5.4.5 Occupants
turkoglu_info_df['Occupants'] = turkoglu_info_df.apply(calculate_occupants, axis=1)

# 5.4.6 Modifying the dataframe
turkoglu_info_df = turkoglu_info_df.drop('Building', axis=1)

# 5.4.7 Damage classification
i_list_turkoglu_damage = list(range(1, len(turkoglu_collapsed) + 1))
coordinates_turkoglu_damage_df = pd.DataFrame({"Building Name": i_list_turkoglu_damage, "Damage Level": turkoglu_collapsed['dmg'], 'Geometry': turkoglu_collapsed['geometry']})

# 1. Geographical location of each building
# Manually obtain the coordinates (latitude and longitude) for the address
latitude_turkoglu = 37.384735
longitude_turkoglu = 36.847980
# Use the coordinates to retrieve OSM data
buildings_turkoglu = ox.features.features_from_point((latitude_turkoglu, longitude_turkoglu), tags, dist=3610)

# 2. Damage Data
# Read Turkoglu colapsed buildings 
file_turkoglu =  Own_filepath + r"\Datasets cities in turkey\new_turkoglu_collapsed.gpkg"
turkoglu_collapsed = gpd.read_file(file_turkoglu)

# Assuming 'buildings_turkoglu' and 'turkoglu_collapsed' are GeoDataFrames with the required data
damaged_building_coordinates_turkey = extract_damaged_building_coordinates(buildings_turkoglu, turkoglu_collapsed)
damaged_building_coordinates_centroids = extract_damaged_building_coordinates_with_centroids(buildings_turkoglu, turkoglu_collapsed)

# From UTM to long/latitude -> Change the coordinate system
myProj = Proj("+proj=utm +zone=37 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")
# The next cell uses the object created (myProj) to convert the UTM positions to longitude and latitude. That is why we need the inverse keyword.
lon, lat = myProj(damaged_building_coordinates_centroids.centroid.apply(lambda p: p.x).values, damaged_building_coordinates_centroids.centroid.apply(lambda p: p.y).values, inverse=True)

damage_turkoglu_df = pd.DataFrame({'Building Name': damaged_building_coordinates_centroids['building'].tolist(), 'Damage Level': damaged_building_coordinates_centroids['dmg'], 'Longitude': np.around(lat,9), 'Latitude': np.around(lon,9)})

# Check overlap between the two tables, overlap is based on the coordinates
overlap_damage_coordinates_turkoglu = turkoglu_info_df.merge(damage_turkoglu_df, on=['Longitude', 'Latitude'], how='outer')
overlap_damage_coordinates_turkoglu = overlap_damage_coordinates_turkoglu[overlap_damage_coordinates_turkoglu['Construction Type'].notna()]

# Since the damage dataset does not show the undamaged data, there was a difference in the amount of buildings
overlap_damage_coordinates_turkoglu['Damage Level'] = overlap_damage_coordinates_turkoglu['Damage Level'].fillna(0)
overlap_damage_coordinates_turkoglu.drop_duplicates(subset=['Longitude', 'Latitude'], keep='first', inplace=True)

overlap_damage_coordinates_turkoglu.rename(columns={'Building Name_x': 'Building Name'}, inplace=True)
overlap_damage_coordinates_turkoglu = overlap_damage_coordinates_turkoglu.drop('Building Name_y', axis=1)

# 5.4.8. Save the output
Total_df_turkoglu_info = overlap_damage_coordinates_turkoglu[['Building Name', 'Damage Level', 'Longitude', 'Latitude', 'Area (m²)', 'Number of Stories', 'Occupants', 'Construction Year', 'Construction Type', 'Hazardous materials']]
Total_df_turkoglu_info.to_excel(r"C:\Users\Véronique\OneDrive\Documenten\Master bouwkunde\Leerjaar 4\CORE\09_Back-End\Dataframes\Turkoglu\Df_Turkoglu.xlsx", index=False)

gpf_turkoglu = gpd.GeoDataFrame(Total_df_turkoglu_info, geometry=gpd.points_from_xy(Total_df_turkoglu_info['Longitude'], Total_df_turkoglu_info['Latitude']))
geojson_filepath = r"C:\Users\Véronique\OneDrive\Documenten\Master bouwkunde\Leerjaar 4\CORE\09_Back-End\Dataframes\Turkoglu\Df_Turkoglu.geojson"
gpf_turkoglu.to_file(geojson_filepath, driver='GeoJSON')