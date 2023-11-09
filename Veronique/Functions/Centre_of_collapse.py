#Imports for Geopandas
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box

#Imports for statistical analysis
from shapely.ops import unary_union
from matplotlib.patches import Ellipse

def plot_damage_centre(building_gdf, city_name, save=False, filepath=None, xlim=None, ylim=None, show=False, close=True):
    """
    Explanation: 
    Visualize the center of damage for buildings in a city, specifically those with damage levels between 3 and 4

    Parameters: 
    building_gdf (GeoDataFrame): A GeoDataFrame containing building data, 'dmg' column
    city_name (str): A string representing the name of the city
    save (bool, optional): If set to True, the plot will be saved as an image.
    filepath (str, optional): If save is set to True, this parameter should specify the file path where the plot image will be saved. 
    xlim (tuple, optional): A tuple specifying the x-axis limits of the plot. 
    ylim (tuple, optional): A tuple specifying the y-axis limits of the plot. 
    show (bool, optional): If set to True, the plot will be displayed.
    close (bool, optional): If set to True, the plot window will be closed after displaying the plot.

    Returns: 
    Generates a plot to visualize the center of damage in the specified city.
    """
    # Filter buildings with damage levels between 3 and 4
    filtered_buildings = building_gdf[(building_gdf['dmg'] >= 3) & (building_gdf['dmg'] <= 4)]

    if not filtered_buildings.empty:
        
        # Calculate the centroid of the filtered buildings
        centroid = filtered_buildings.unary_union.centroid

        # Create a GeoDataFrame for the centroid
        centroid_gdf = gpd.GeoDataFrame({'geometry': [centroid]}, crs=building_gdf.crs)

        # Create an Ellipse to represent the moving ellipse
        avg_x, avg_y = centroid.x, centroid.y
        width = 10  # Adjust width as needed
        height = 0.005  # Adjust height as needed

        # Create the ellipse
        ellipse = Ellipse(
            (avg_x, avg_y),
            width,
            height,
            edgecolor='red',  # Set the edge color to red
            fill=False,  # Do not fill the ellipse
            alpha=0.5  # Adjust opacity as needed
        )

        # Plot the buildings, the moving ellipse, and the centroid
        ax = building_gdf.plot(figsize=(10, 10), alpha=0.5)
        ax.add_patch(ellipse)  # Add the ellipse to the plot
        centroid_gdf.plot(ax=ax, color='red', markersize=6000, marker='o', alpha=0.4)

        # Set plot title and labels
        ax.set_title(f'Centre of damage {city_name}')

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
    else:
        print(f'No buildings with damage levels between 3 and 4 in {city_name}.')

