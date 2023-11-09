import geopandas as gpd
import numpy as np

def process_and_save_to_gpkg(input_file, output_file):
    # Read the input GeoPackage file
    data = gpd.read_file(input_file)

    # Process the data
    data['dmg'] = np.around(data['damage_pct'] * 4, 1)
    data['Building'] = data['id'] + 1

    # Select relevant columns
    processed_data = data[['Building', 'dmg', 'geometry']]

    # Create a GeoDataFrame
    gdf = gpd.GeoDataFrame(processed_data, geometry='geometry')

    # Save the GeoDataFrame to the output GeoPackage file
    gdf.to_file(output_file, driver='GPKG')

# Define the input and output file paths
marash_input_file = 'C:/Users/Véronique/OneDrive/Documenten/Master bouwkunde/Leerjaar 4/CORE/Datasets cities in turkey/marash_msft_damage_2_9_2023.gpkg'
marash_output_file = 'C:/Users/Véronique/OneDrive/Documenten/Master bouwkunde/Leerjaar 4/CORE/Datasets cities in turkey/new_marash_collapsed.gpkg'

turkoglu_input_file = 'C:/Users/Véronique/OneDrive/Documenten/Master bouwkunde/Leerjaar 4/CORE/Datasets cities in turkey/turkoglu_msft_damage_2_9_2023.gpkg'
turkoglu_output_file = 'C:/Users/Véronique/OneDrive/Documenten/Master bouwkunde/Leerjaar 4/CORE/Datasets cities in turkey/new_turkoglu_collapsed.gpkg'

# Process and save the "Marash" data
process_and_save_to_gpkg(marash_input_file, marash_output_file)

# Process and save the "Turkoglu" data
process_and_save_to_gpkg(turkoglu_input_file, turkoglu_output_file)