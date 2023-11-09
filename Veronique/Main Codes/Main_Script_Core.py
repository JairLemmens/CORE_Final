# Imports for Openstreetmapry
import osmnx as ox
# Imports for Geopandas
import geopandas as gpd
from shapely.geometry import shape
from shapely.geometry import Polygon
# Imports for statistical analysis
import pandas as pd

# Imports for functions
# 1 Geographical location
from Coordinates_footprints_turkey import gen_building_coordinates_turkey
# 2 Damage classification
# 2.1 Location damaged buildings
from Coordinates_damaged_buildings import extract_damaged_building_coordinates, extract_damaged_building_coordinates_with_centroids
# 2.2 Collapsed and damaged map
from Damaged_map_collapsed_buildings import gen_map_damage_buildings
# 2.3 Visulisation damage and collapse
from Visualisation_Damaged_Map import plot_building_damage_pattern_visualisation
# 2.4 Damage shape
from Damage_Shape import create_damage_shapes, plot_building_damage_pattern_with_clustering
# 2.5 Centre of damage
from Centre_of_collapse import plot_damage_centre
# 2.6 Direction of damage map
from Direction_collapse_map import calculate_building_orientation, plot_building_damage_pattern_with_orientation
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
# Download building footprints for a specific place
# Gaziantep, Islahiye
buildings_islahiye = ox.features.features_from_address('Şht. Zafer Yılmaz Cd. 15, 27800 İslahiye', tags, dist=10418) 
# To plot all the buildings in islahiye
ox.plot_footprints(buildings_islahiye, save=True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Open Street Maps\Islahiye.jpeg", show=False, close=True)

# Extract coordinates (centroid) of each building
# Call the function to generate coordinates for buildings in Islahiye
result_coordinates_turkey_islahiye = gen_building_coordinates_turkey(buildings_islahiye)
# Print the coordinates
txt_file_footprints_islahiye = Own_filepath + r"\09_Back-End\txt_files\Islahiye\Coordinates of the footprints.txt"
# Write the coordinates to the text file 
with open(txt_file_footprints_islahiye, "w") as f:
    for i, row in result_coordinates_turkey_islahiye.iterrows():
        f.write(f"Building {i+1} Coordinate: {row['Latitude']} {row['Longitude']}\n")
f.close()

# Gaziantep, Nurdagi
buildings_nurdagi = ox.features.features_from_address('Cengiz Topel Cd. 1, 27840 Nurdağı/Gaziantep', tags, dist=4100) 
# To plot all the buildings in nurdagi
ox.plot_footprints(buildings_nurdagi, save=True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Open Street Maps\Nurdagi.jpeg", show=False, close=True)

# Extract coordinates (centroid) of each building
# Call the function to generate coordinates for buildings in Nurdagi
result_coordinates_turkey_nurdagi = gen_building_coordinates_turkey(buildings_nurdagi)
# Print the coordinates
txt_file_footprints_nurdagi = Own_filepath + r"\09_Back-End\txt_files\Nurdagi\Coordinates of the footprints.txt"
# Write the coordinates to the text file 
with open(txt_file_footprints_nurdagi, "w") as f:
    for i, row in result_coordinates_turkey_nurdagi.iterrows():
        f.write(f"Building {i+1} Coordinate: {row['Latitude']} {row['Longitude']}\n")
f.close()

# Kahramanmaraş, Marash
# Manually obtain the coordinates (latitude and longitude) for the address
latitude_marash = 37.574038
longitude_marash = 36.920378 

# Use the coordinates to retrieve OSM data
buildings_marash = ox.features.features_from_point((latitude_marash, longitude_marash), tags, dist=3027)
# To plot all the buildings in marash
ox.plot_footprints(buildings_marash, save=True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Open Street Maps\Marash.jpeg", show=False, close=True)

# Extract coordinates (centroid) of each building
# Call the function to generate coordinates for buildings in Marash
result_coordinates_turkey_marash = gen_building_coordinates_turkey(buildings_marash)
# Print the coordinates
# Print the coordinates
txt_file_footprints_marash =Own_filepath + r"\09_Back-End\txt_files\Marash\Coordinates of the footprints.txt"
# Write the coordinates to the text file 
with open(txt_file_footprints_marash, "w") as f:
    for i, row in result_coordinates_turkey_marash.iterrows():
        f.write(f"Building {i+1} Coordinate: {row['Latitude']} {row['Longitude']}\n")
f.close()

# Kahramanmaraş, Turkoglu
# Manually obtain the coordinates (latitude and longitude) for the address
latitude_turkoglu = 37.384735
longitude_turkoglu = 36.847980

# Use the coordinates to retrieve OSM data
buildings_turkoglu = ox.features.features_from_point((latitude_turkoglu, longitude_turkoglu), tags, dist=3610)

# To plot all the buildings in turkoglu
ox.plot_footprints(buildings_turkoglu, save=True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Open Street Maps\Turkoglu.jpeg", show=False, close=True)

# Extract coordinates (centroid) of each building
# Call the function to generate coordinates for buildings in Marash
result_coordinates_turkey_turkoglu = gen_building_coordinates_turkey(buildings_turkoglu)
# Print the coordinates
txt_file_footprints_turkoglu =Own_filepath + r"\09_Back-End\txt_files\Turkoglu\Coordinates of the footprints.txt"
# Write the coordinates to the text file 
with open(txt_file_footprints_turkoglu, "w") as f:
    for i, row in result_coordinates_turkey_turkoglu.iterrows():
        f.write(f"Building {i+1} Coordinate: {row['Latitude']} {row['Longitude']}\n")
f.close()

# 2. Damage Classification
# Read the data for collapsed buildings
# Read cities in Turkey --> Geopackage Format
# Read islahiye colapsed buildings
file_islahiye = Own_filepath + r"\Datasets cities in turkey\islahiye_osm_damage_2_7_2023.gpkg"
islahiye_collapsed = gpd.read_file(file_islahiye)

# 2.1 Location damaged buildings islahiye
extract_damaged_building_coordinates(buildings_islahiye, islahiye_collapsed)
# Assuming 'buildings_islahiye' and 'islahiye_collapsed' are GeoDataFrames with the required data
damaged_building_coordinates_islahiye = extract_damaged_building_coordinates(buildings_islahiye, islahiye_collapsed)
# Print the coordinates of the damaged buildings
txt_file_footprints_damaged_islahiye = Own_filepath + r"\09_Back-End\txt_files\Islahiye\Coordinates of the damaged footprints.txt"
# Write the coordinates of the damaged polygons of Islahiye to the text file 
with open(txt_file_footprints_damaged_islahiye, "w") as f:
    for i, row in damaged_building_coordinates_islahiye.iterrows():
        f.write(f"Building {i+1} Damage: {row['dmg']} geometry{row['geometry']}\n")
f.close()

damaged_building_centroids_coordinates_islahiye = extract_damaged_building_coordinates_with_centroids(buildings_islahiye, islahiye_collapsed)
# Print the centroids coordinates of the damaged buildings

txt_file_footprints_centroids_damaged_islahiye = Own_filepath + r"\09_Back-End\txt_files\Islahiye\Coordinates of the centroids of the damaged footprints.txt"
# Write the centroids coordinates of the damaged polygons of Islahiye to the text file 
with open(txt_file_footprints_centroids_damaged_islahiye, "w") as f:
    for i, row in damaged_building_centroids_coordinates_islahiye.iterrows():
        f.write(f"Building {i+1} Damage: {row['dmg']} centroid: {row['geometry'].centroid}\n")
f.close()

# 2.2 Collapsed and damaged map
gen_map_damage_buildings(islahiye_collapsed, name='Islahiye', save=True, filepath= Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage\Islahiye.jpeg", xlim=(288000, 291000), ylim=(4098500, 4102000), show=False, close=True)

# 2.3 Visulisation damage and collapse
plot_building_damage_pattern_visualisation(islahiye_collapsed, "Islahiye", save=True, filepath= Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage visualisation\Islahiye.jpeg", xlim=(288000, 291000), ylim=(4098500, 4102000), show=False, close=True)

# 2.4 Damage shape
# Set the highest and middle damage thresholds
# This is a parameter which will be used for all the four cities, adjust when necessary 
highest_damage_threshold = 4
medium_damage_threshold = 3
middle_damage_threshold = 2

result_df, highest_damage_shape, middle_damage_shape = create_damage_shapes(islahiye_collapsed, highest_damage_threshold, middle_damage_threshold)

txt_file_damage_shape_islahiye = Own_filepath + r"\09_Back-End\txt_files\Islahiye\Damage Shape.txt"

# Save the result_df DataFrame to a custom text file
with open(txt_file_damage_shape_islahiye, "w") as f:
    f.write("Damage Type, Geometry \n")
    # Access the "Building Type" from the DataFrame
    Damage_Type = result_df[['Damage Type']]
    # Use highest_damage_shape directly as the geometry
    geometry = highest_damage_shape
    f.write(f"{Damage_Type}, {geometry.geoms}\n")
f.close()

plot_building_damage_pattern_with_clustering(islahiye_collapsed, "Islahiye", medium_damage_threshold, middle_damage_threshold,save=True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Collapse cluster\Islahiye.jpeg", xlim=(288000, 291000), ylim=(4098500, 4102000), show=False, close=True)

# 2.5 Centre of the damage
plot_damage_centre(islahiye_collapsed, "Islahiye", save = True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage centre\Islahiye.jpeg", xlim=(288000, 291000), ylim=(4098500, 4102000), show=False, close=True)

# 2.6 Direction of damage map
# Selecting the geometry column
building_footprint_islahiye = damaged_building_coordinates_islahiye["geometry"]

txt_file_islahiye_orientation = Own_filepath + r"\09_Back-End\txt_files\Islahiye\Building orientation.txt"
f = open(txt_file_islahiye_orientation, "w")

for idx, geom in enumerate(damaged_building_coordinates_islahiye['geometry']):
    if not geom.is_empty and isinstance(geom, Polygon):
        orientation = calculate_building_orientation(geom)
        if orientation is not None:
            f.write(f"Building {idx} orientation: {orientation:.2f} degrees. \n")
        else:
            f.write(f"Orientation for Building {idx} is None. \n")
    else:
        f.write(f"Invalid or empty geometry for Building {idx}. \n")
f.close()

# normalize angle figure out how to put here
plot_building_damage_pattern_with_orientation(islahiye_collapsed, "Islahiye", save = True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage orientation\Islahiye.jpeg", xlim=(288000, 291000), ylim=(4098500, 4102000), show=False, close=True)

# Read Nurdagi colapsed buildings
file_nurdagi = Own_filepath + r"\Datasets cities in turkey\nurdagi_msft_damage_2_7_2023.gpkg"
nurdagi_collapsed = gpd.read_file(file_nurdagi)

# 2.1 Location damaged buildings nurdagi
extract_damaged_building_coordinates(buildings_nurdagi, nurdagi_collapsed)
# Assuming 'buildings_nurdagi' and 'nurdagi_collapsed' are GeoDataFrames with the required data
damaged_building_coordinates_nurdagi = extract_damaged_building_coordinates(buildings_nurdagi, nurdagi_collapsed)
# Print the coordinates of the damaged buildings
txt_file_footprints_damaged_nurdagi = Own_filepath + r"\09_Back-End\txt_files\Nurdagi\Coordinates of the damaged footprints.txt"
# Write the coordinates of the damaged polygons of Nurdagi to the text file 
with open(txt_file_footprints_damaged_nurdagi, "w") as f:
    for i, row in damaged_building_coordinates_nurdagi.iterrows():
        f.write(f"Building {i+1} Damage: {row['dmg']} geometry{row['geometry']}\n")
f.close()

damaged_building_centroids_coordinates_nurdagi = extract_damaged_building_coordinates_with_centroids(buildings_nurdagi, nurdagi_collapsed)
# Print the centroids coordinates of the damaged buildings

txt_file_footprints_centroids_damaged_nurdagi = Own_filepath + r"\09_Back-End\txt_files\Nurdagi\Coordinates of the centroids of the damaged footprints.txt"
# Write the centroids coordinates of the damaged polygons of Nurdagi to the text file 
with open(txt_file_footprints_centroids_damaged_nurdagi, "w") as f:
    for i, row in damaged_building_centroids_coordinates_nurdagi.iterrows():
        f.write(f"Building {i+1} Damage: {row['dmg']} centroid: {row['geometry'].centroid}\n")
f.close()

# 2.2 Collapsed and damaged map
gen_map_damage_buildings(nurdagi_collapsed, name='Nurdagi', save=True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage\Nurdagi.jpeg", xlim=(297000, 302000), ylim=(4116000, 4118700), show=False, close=True)

# 2.3 Visulisation damage and collapse
plot_building_damage_pattern_visualisation(nurdagi_collapsed, "Nurdagi", save=True, filepath= Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage visualisation\Nurdagi.jpeg",xlim=(297000, 302000), ylim=(4116000, 4118700), show=False, close=True )

# 2.4 Damage shape
highest_damage_threshold = 4
medium_damage_threshold = 3
middle_damage_threshold = 2

result_df, highest_damage_shape, middle_damage_shape = create_damage_shapes(nurdagi_collapsed, highest_damage_threshold, middle_damage_threshold)

txt_file_damage_shape_nurdagi = Own_filepath + r"\09_Back-End\txt_files\Nurdagi\Damage Shape.txt"

# Save the result_df DataFrame to a custom text file
with open(txt_file_damage_shape_nurdagi, "w") as f:
    f.write("Damage Type, Geometry \n")
    # Access the "Building Type" from the DataFrame
    Damage_Type = result_df[['Damage Type']]
    # Use highest_damage_shape directly as the geometry
    geometry = highest_damage_shape
    f.write(f"{Damage_Type}, {geometry.geoms}\n")
f.close()

plot_building_damage_pattern_with_clustering(nurdagi_collapsed, "Nurdagi", medium_damage_threshold, middle_damage_threshold,save=True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Collapse cluster\Nurdagi.jpeg", xlim=(297000, 302000), ylim=(4116000, 4118700), show=False, close=True)

# 2.5 Centre of the damage
plot_damage_centre(nurdagi_collapsed, "Nurdagi",save = True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage centre\Nurdagi.jpeg", xlim=(297000, 302000), ylim=(4116000, 4118700), show=False, close=True)

# 2.6 Direction of damage map
# Selecting the geometry column
building_footprint_nurdagi = damaged_building_coordinates_nurdagi["geometry"]

txt_file_nurdagi_orientation = Own_filepath + r"\09_Back-End\txt_files\Nurdagi\Building orientation.txt"
f = open(txt_file_nurdagi_orientation, "w")

for idx, geom in enumerate(damaged_building_coordinates_nurdagi['geometry']):
    if not geom.is_empty and isinstance(geom, Polygon):
        orientation = calculate_building_orientation(geom)
        if orientation is not None:
            f.write(f"Building {idx} orientation: {orientation:.2f} degrees. \n")
        else:
            f.write(f"Orientation for Building {idx} is None. \n")
    else:
        f.write(f"Invalid or empty geometry for Building {idx}. \n")
f.close()

#normalize angle figure out how to put here
plot_building_damage_pattern_with_orientation(nurdagi_collapsed, "Nurdagi", save = True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage orientation\Nurdagi.jpeg", xlim=(297000, 302000), ylim=(4116000, 4118700), show=False, close=True)

# Read Marash colapsed buildings
file_marash = Own_filepath + r"\Datasets cities in turkey\new_marash_collapsed.gpkg"
marash_collapsed = gpd.read_file(file_marash)

# 2.1 Location damaged buildings marash
extract_damaged_building_coordinates(buildings_marash, marash_collapsed)
# Assuming 'buildings_marash' and 'marash_collapsed' are GeoDataFrames with the required data
damaged_building_coordinates_marash = extract_damaged_building_coordinates(buildings_marash, marash_collapsed)
# Print the coordinates of the damaged buildings
txt_file_footprints_damaged_marash = Own_filepath + r"\09_Back-End\txt_files\Marash\Coordinates of the damaged footprints.txt"
# Write the coordinates of the damaged polygons of Marash to the text file 
with open(txt_file_footprints_damaged_marash, "w") as f:
    for i, row in damaged_building_coordinates_marash.iterrows():
        f.write(f"Building {i+1} Damage: {row['dmg']} geometry{row['geometry']}\n")
f.close()

damaged_building_centroids_coordinates_marash = extract_damaged_building_coordinates_with_centroids(buildings_marash, marash_collapsed)
# Print the centroids coordinates of the damaged buildings

txt_file_footprints_centroids_damaged_marash = Own_filepath + r"\09_Back-End\txt_files\Marash\Coordinates of the centroids of the damaged footprints.txt"
# Write the centroids coordinates of the damaged polygons of Marash to the text file 
with open(txt_file_footprints_centroids_damaged_marash, "w") as f:
    for i, row in damaged_building_centroids_coordinates_marash.iterrows():
        f.write(f"Building {i+1} Damage: {row['dmg']} centroid: {row['geometry'].centroid}\n")
f.close()

# 2.2 Collapsed and damaged map
gen_map_damage_buildings(marash_collapsed, name='Marash', save=True, filepath= Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage\Marash.jpeg", xlim=(310000, 320000), ylim=(4158000, 4163000), show=False, close=True)

# 2.3 Visulisation damage and collapse
plot_building_damage_pattern_visualisation(marash_collapsed, "Marash", save=True, filepath= Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage visualisation\Marash.jpeg",xlim=(310000, 320000), ylim=(4158000, 4163000), show=False, close=True )

# 2.4 Damage shape
highest_damage_threshold = 4
medium_damage_threshold = 3
middle_damage_threshold = 2

result_df, highest_damage_shape, middle_damage_shape = create_damage_shapes(marash_collapsed, highest_damage_threshold, middle_damage_threshold)

txt_file_damage_shape_marash = Own_filepath + r"\09_Back-End\txt_files\Marash\Damage Shape.txt"

# Save the result_df DataFrame to a custom text file
with open(txt_file_damage_shape_marash, "w") as f:
    f.write("Damage Type, Geometry \n")
    # Access the "Building Type" from the DataFrame
    Damage_Type = result_df[['Damage Type']]
    # Use highest_damage_shape directly as the geometry
    geometry = highest_damage_shape
    f.write(f"{Damage_Type}, {geometry.geoms}\n")
f.close()

plot_building_damage_pattern_with_clustering(marash_collapsed, "Marash", medium_damage_threshold, middle_damage_threshold,save=True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Collapse cluster\Marash.jpeg", xlim=(310000, 320000), ylim=(4158000, 4163000), show=False, close=True)

# 2.5 Centre of the damage
plot_damage_centre(marash_collapsed, "Marash", save = True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage centre\Marash.jpeg", xlim=(310000, 320000), ylim=(4158000, 4163000), show=False, close=True)

# 2.6 Direction of damage map
# Selecting the geometry column
building_footprint_marash = damaged_building_coordinates_marash["geometry"]

txt_file_marash_orientation = Own_filepath + r"\09_Back-End\txt_files\Marash\Building orientation.txt"
f = open(txt_file_marash_orientation, "w")

for idx, geom in enumerate(damaged_building_coordinates_marash['geometry']):
    if not geom.is_empty and isinstance(geom, Polygon):
        orientation = calculate_building_orientation(geom)
        if orientation is not None:
            f.write(f"Building {idx} orientation: {orientation:.2f} degrees. \n")
        else:
            f.write(f"Orientation for Building {idx} is None. \n")
    else:
        f.write(f"Invalid or empty geometry for Building {idx}. \n")
f.close()

#normalize angle figure out how to put here
plot_building_damage_pattern_with_orientation(marash_collapsed, "Marash", save = True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage orientation\Marash.jpeg", xlim=(310000, 320000), ylim=(4158000, 4163000), show=False, close=True)

# Read Turkoglu colapsed buildings 
file_turkoglu =  Own_filepath + r"\Datasets cities in turkey\new_turkoglu_collapsed.gpkg"
turkoglu_collapsed = gpd.read_file(file_turkoglu)

# 2.1 Location damaged buildings turkoglu
extract_damaged_building_coordinates(buildings_turkoglu, turkoglu_collapsed)
# Assuming 'buildings_turkoglu' and 'turkoglu_collapsed' are GeoDataFrames with the required data
damaged_building_coordinates_turkoglu = extract_damaged_building_coordinates(buildings_turkoglu, turkoglu_collapsed)
# Print the coordinates of the damaged buildings
txt_file_footprints_damaged_turkoglu = Own_filepath + r"\09_Back-End\txt_files\Turkoglu\Coordinates of the damaged footprints.txt"
# Write the coordinates of the damaged polygons of turkoglu to the text file 
with open(txt_file_footprints_damaged_turkoglu, "w") as f:
    for i, row in damaged_building_coordinates_turkoglu.iterrows():
        f.write(f"Building {i+1} Damage: {row['dmg']} geometry{row['geometry']}\n")
f.close()

damaged_building_centroids_coordinates_turkoglu = extract_damaged_building_coordinates_with_centroids(buildings_turkoglu, turkoglu_collapsed)
# Print the centroids coordinates of the damaged buildings

txt_file_footprints_centroids_damaged_turkoglu = Own_filepath + r"\09_Back-End\txt_files\Turkoglu\Coordinates of the centroids of the damaged footprints.txt"
# Write the centroids coordinates of the damaged polygons of turkoglu to the text file 
with open(txt_file_footprints_centroids_damaged_turkoglu, "w") as f:
    for i, row in damaged_building_centroids_coordinates_turkoglu.iterrows():
        f.write(f"Building {i+1} Damage: {row['dmg']} centroid: {row['geometry'].centroid}\n")
f.close()

# vanaf hier zijn er errors na lezen!!!
# 2.2 Collapsed and damaged map
gen_map_damage_buildings(turkoglu_collapsed, name='Turkoglu', save=True, filepath= Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage\Turkoglu.jpeg", xlim=(308500, 311500), ylim=(4138000, 4141000), show=False, close=True)

# 2.3 Visulisation damage and collapse
plot_building_damage_pattern_visualisation(turkoglu_collapsed, "Turkoglu", save=True, filepath= Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage visualisation\Turkoglu.jpeg", xlim=(308500, 311500), ylim=(4138000, 4141000), show=False, close=True )

# 2.4 Damage shape
highest_damage_threshold = 4
medium_damage_threshold = 3
middle_damage_threshold = 2

result_df, highest_damage_shape, middle_damage_shape = create_damage_shapes(turkoglu_collapsed, highest_damage_threshold, middle_damage_threshold)

txt_file_damage_shape_turkoglu = Own_filepath + r"\09_Back-End\txt_files\Turkoglu\Damage Shape.txt"

# Save the result_df DataFrame to a custom text file
with open(txt_file_damage_shape_turkoglu, "w") as f:
    f.write("Damage Type, Geometry \n")
    # Access the "Building Type" from the DataFrame
    Damage_Type = result_df[['Damage Type']]
    # Use highest_damage_shape directly as the geometry
    geometry = highest_damage_shape
    f.write(f"{Damage_Type}, {geometry.geoms}\n")
f.close()

plot_building_damage_pattern_with_clustering(turkoglu_collapsed, "Turkoglu", medium_damage_threshold, middle_damage_threshold,save=True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Collapse cluster\Turkoglu.jpeg", xlim=(308500, 311500), ylim=(4138000, 4141000), show=False, close=True)

# 2.5 Centre of the damage
plot_damage_centre(turkoglu_collapsed, "Turkoglu", save = True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage centre\Turkoglu.jpeg", xlim=(308500, 311500), ylim=(4138000, 4141000), show=False, close=True)

# 2.6 Direction of damage map
# Selecting the geometry column
building_footprint_turkoglu = damaged_building_coordinates_turkoglu["geometry"]

txt_file_turkoglu_orientation = Own_filepath + r"\09_Back-End\txt_files\Turkoglu\Building orientation.txt"
f = open(txt_file_turkoglu_orientation, "w")

for idx, geom in enumerate(damaged_building_coordinates_turkoglu['geometry']):
    if not geom.is_empty and isinstance(geom, Polygon):
        orientation = calculate_building_orientation(geom)
        if orientation is not None:
            f.write(f"Building {idx} orientation: {orientation:.2f} degrees. \n")
        else:
            f.write(f"Orientation for Building {idx} is None. \n")
    else:
        f.write(f"Invalid or empty geometry for Building {idx}. \n")
f.close()

#normalize angle figure out how to put here
plot_building_damage_pattern_with_orientation(turkoglu_collapsed, "Turkoglu", save = True, filepath=Own_filepath + r"\09_Back-End\Created Maps\Damage Maps\Damage orientation\Turkoglu.jpeg", xlim=(308500, 311500), ylim=(4138000, 4141000), show=False, close=True)

# 3. Area
# 3.1 Area islahiye buildings
# Call the function and store the result
result_table_area_islahiye = gen_area_buildings_turkey(buildings_islahiye)
result_table_area_islahiye.to_excel(Own_filepath + r"\09_Back-End\Area excel files\Results_table_area_Islahiye.xlsx")

# 3.2 Area nurdagi buildings
# Call the function and store the result
result_table_area_nurdagi = gen_area_buildings_turkey(buildings_nurdagi)
result_table_area_nurdagi.to_excel(Own_filepath + r"\09_Back-End\Area excel files\Results_table_area_Nurdagi.xlsx")

# 3.3 Area marash buildings
# Call the function and store the result
result_table_area_marash = gen_area_buildings_turkey(buildings_marash)
result_table_area_marash.to_excel(Own_filepath + r"\09_Back-End\Area excel files\Results_table_area_Marash.xlsx")

# 3.4 Area turkoglu buildings
# Call the function and store the result
result_table_area_turkoglu = gen_area_buildings_turkey(buildings_turkoglu)
result_table_area_turkoglu.to_excel(Own_filepath + r"\09_Back-End\Area excel files\Results_table_area_Turkoglu.xlsx")

# 4 Building properties
# reading excel file 
# 4.1 Islahiye Generate building data
building_data_islahiye = gen_buildingdata_gaziantep(count=13215)
df = pd.DataFrame(building_data_islahiye)
df.to_excel(Own_filepath + r"\09_Back-End\Building_properties\Building_properties_Islahiye.xlsx")

# 4.2 Nurdagi Generate building data
building_data_nurdagi = gen_buildingdata_gaziantep(count=4537)
df = pd.DataFrame(building_data_nurdagi)
df.to_excel(Own_filepath + r"\09_Back-End\Building_properties\Building_properties_Nurdagi.xlsx")

# 4.3 Marash Generate building data
building_data_marash = gen_buildingdata_kahramanmaraş(count=40375)
df = pd.DataFrame(building_data_marash)
df.to_excel(Own_filepath + r"\09_Back-End\Building_properties\Building_properties_Marash.xlsx")

# 4.4 Turkoglu Generate building data
building_data_turkoglu = gen_buildingdata_kahramanmaraş(count=3816)
df = pd.DataFrame(building_data_turkoglu)
df.to_excel(Own_filepath + r"\09_Back-End\Building_properties\Building_properties_Turkoglu.xlsx")


