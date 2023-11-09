#Imports for Geopandas analysis
import geopandas as gpd
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import pandas as pd

#Imports for statistical analysis
from shapely.geometry import shape
from shapely.geometry import box

from Direction_collapse_map import calculate_building_orientation
from Direction_collapse_map import normalize_angle

# Define a function to create damage shapes
def create_damage_shapes(building_gdf, highest_damage_threshold, middle_damage_threshold):
    """
    Explanation: 
    Create polygons for buildings with specified damage levels.

    Parameters:
    building_gdf (GeoDataFrame): Building data with 'dmg' column.
    highest_damage_threshold (int): Highest damage level.
    middle_damage_threshold (int): Middle damage level.

    Returns:
    Two polygons.
    """
    building_gdf['dmg'] = building_gdf['dmg'].astype(float)

    highest_damage_buildings = building_gdf[(building_gdf['dmg'] >= highest_damage_threshold) & (building_gdf['dmg'] <= 4)]
    middle_damage_buildings = building_gdf[(building_gdf['dmg'] >= middle_damage_threshold) & (building_gdf['dmg'] < highest_damage_threshold)]

    highest_damage_shape = Polygon()
    for geom in highest_damage_buildings['geometry']:
        if not geom.is_empty:
            if geom.geom_type == 'Polygon':
                highest_damage_shape = highest_damage_shape.union(geom)
            elif geom.geom_type == 'MultiPolygon':
                for subgeom in geom.geoms:
                    highest_damage_shape = highest_damage_shape.union(subgeom)

    middle_damage_shape = Polygon()
    for geom in middle_damage_buildings['geometry']:
        if not geom.is_empty:
            if geom.geom_type == 'Polygon':
                middle_damage_shape = middle_damage_shape.union(geom)
            elif geom.geom_type == 'MultiPolygon':
                for subgeom in geom.geoms:
                    middle_damage_shape = middle_damage_shape.union(subgeom)
    
    # Create a DataFrame to store the results
    result_df = pd.DataFrame({
        'Damage Type': ['Highest Damage', 'Middle Damage'],
        'Geometry': [highest_damage_shape, middle_damage_shape]
    })
    return result_df, highest_damage_shape, middle_damage_shape

def plot_building_damage_pattern_with_clustering(building_gdf, city_name, medium_damage_threshold, middle_damage_threshold, save=False, filepath=None, xlim=None, ylim=None, show=True, close=True):
    """
    Plot building damage patterns in a specific city with clustering and save the plot as an image if desired.

    Parameters:
    - building_gdf (GeoDataFrame): A GeoDataFrame containing building geometries and damage information.
    - city_name (str): The name of the city for which the damage patterns are to be plotted.
    - medium_damage_threshold (float): The threshold for medium damage classification.
    - middle_damage_threshold (float): The threshold for middle damage classification.
    - save (bool, optional): If True, the plot will be saved as an image. Default is False.
    - filepath (str, optional): The file path where the image will be saved if 'save' is True. Default is None.
    - xlim (tuple, optional): A tuple specifying the x-axis limits for the plot (e.g., (xmin, xmax)). Default is None.
    - ylim (tuple, optional): A tuple specifying the y-axis limits for the plot (e.g., (ymin, ymax)). Default is None.
    - show (bool, optional): If True, the plot will be displayed. Default is True.
    - close (bool, optional): If True, the plot window will be closed after displaying. Default is True.
    """
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Add a legend
    legend_title = f'Collapse cluster_{city_name}'
   
    # Set the aspect ratio to a fixed value (e.g., 1)
    ax.set_aspect(1)

    # Create damage shapes with appropriate thresholds
    result_df, medium_damage_shape, middle_damage_shape = create_damage_shapes(building_gdf, medium_damage_threshold, middle_damage_threshold)

    # Merge the polygons into a single polygon
    if not medium_damage_shape.is_empty:
        single_damage_polygon = medium_damage_shape.buffer(0)

        # Create a GeoDataFrame for the single damage polygon
        single_damage_gdf = gpd.GeoDataFrame({'geometry': [single_damage_polygon]})

        # Set the CRS for the GeoDataFrame
        single_damage_gdf.crs = 'epsg:4326'

        # Plot the single damage polygon
        single_damage_gdf.plot(ax=ax, color='red', alpha=0.8)
        
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    # Save the plot as an image if 'save' is True and 'filepath' is provided
    if save and filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

    # Show the plot
    if show:
        plt.show()

    # Close the plot window if 'close' is True
    if close:
        plt.close()

 

