#Imports for Geopandas
import numpy as np
import math
import matplotlib.pyplot as plt
from shapely.geometry import box

#Imports for statistical analysis
from matplotlib.patches import Ellipse
from shapely.affinity import rotate

#imports for functions
from Coordinates_damaged_buildings import extract_damaged_building_coordinates

def calculate_building_orientation(footprint):
    """
    Explanation:
    Calculate the orientation (angle) of a building's footprint.

    Parameters:
    footprint (shapely.geometry.Polygon): Polygon representing the building's footprint.

    Returns:
    float: The orientation angle in degrees.
    """
    # Calculate the minimum rotated rectangle
    min_rect = footprint.minimum_rotated_rectangle

    # Get the coordinates of the four corners of the rectangle
    coords = list(min_rect.exterior.coords)

    # Calculate the angle of the first side of the rectangle with the x-axis
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    angle_rad = math.atan2(y2 - y1, x2 - x1)

    # Convert the angle to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg
  
def normalize_angle(angle):
    """
    Explanation:
    Normalize an angle to be within the range [0, 360) degrees.

    Parameters:
    angle (float): The input angle in degrees.

    Returns:
    float: The normalized angle.
    """
    while angle < 0:
        angle += 360
    while angle >= 360:
        angle -= 360
    return angle

# Define a function to plot building damage pattern with oriented ellipses
def plot_building_damage_pattern_with_orientation(building_gdf, city_name, save=False, filepath=None, xlim=None, ylim=None, show=True, close=True):
    """
    Explanation:
    Plot building damage patterns in a specific city with building orientations and save the plot as an image if desired.

    Parameters:
    - building_gdf (GeoDataFrame): A GeoDataFrame containing building geometries and damage information.
    - city_name (str): The name of the city for which the damage patterns are to be plotted.
    - save (bool, optional): If True, the plot will be saved as an image. Default is False.
    - filepath (str, optional): The file path where the image will be saved if 'save' is True. Default is None.
    - xlim (tuple, optional): A tuple specifying the x-axis limits for the plot (e.g., (xmin, xmax)). Default is None.
    - ylim (tuple, optional): A tuple specifying the y-axis limits for the plot (e.g., (ymin, ymax)). Default is None.
    - show (bool, optional): If True, the plot will be displayed. Default is True.
    - close (bool, optional): If True, the plot window will be closed after displaying. Default is True.

    """
    fig, ax = plt.subplots()

    # Create an empty list for legend labels
    legend_labels = []

    # Create lists to store valid orientations and their corresponding geometries
    valid_orientations = []
    valid_geometries = []

    # Plot the buildings
    building_gdf.plot(ax=ax, column= "dmg", cmap='OrRd', legend=True, legend_kwds={'label': "Damage Levels"})

    # Iterate over each building in the GeoDataFrame
    for idx, row in building_gdf.iterrows():
        color = 'green'  # Default color for undamaged buildings
        if row["dmg"] > 0:
            if row["dmg"] <= 1.0:
                color = 'yellow'  # Light yellow for low damage
            else:
                color = 'red'  # Red for higher damage

            # Check if the geometry is valid (not empty or None)
            if row['geometry'] and not row['geometry'].is_empty:
                # Get the centroid of the geometry
                centroid = row['geometry'].centroid

                # Calculate the building orientation
                orientation = calculate_building_orientation(row['geometry'])

                if orientation is not None:
                    # Normalize the orientation angle to a consistent range (0 to 360 degrees)
                    orientation = normalize_angle(orientation)

                    # Add a corresponding label to the legend
                    legend_labels.append(row["dmg"])

                    # Create an oriented ellipse
                    ellipse = Ellipse(
                        xy=(centroid.x, centroid.y),
                        width=80,
                        height=40,
                        angle=orientation,
                        color=color,
                        alpha=0.5
                    )

                    ax.add_patch(ellipse)

                    # Store valid orientations and geometries
                    valid_orientations.append(orientation)
                    valid_geometries.append(row['geometry'])

    # Calculate the average orientation
    avg_orientation = np.mean(valid_orientations)

    # Calculate the centroid based on the centroids of damaged buildings
    avg_x = np.mean([geom.centroid.x for geom in valid_geometries])
    avg_y = np.mean([geom.centroid.y for geom in valid_geometries])

    # Create the average ellipse
    avg_ellipse = Ellipse(
        xy=(avg_x, avg_y),
        width=200,
        height=80,
        angle=avg_orientation,
        color='blue',
        alpha=0.6
    )
    
    # Create the label with latitude and longitude and add it to the legend
    avg_label = f"Avg Lat: {avg_y:.2f}\nAvg Lon: {avg_x:.2f}\nAvg Orientation: {avg_orientation:.2f} degrees"
    avg_ellipse.set_label(avg_label)
    ax.add_patch(avg_ellipse)

    # Set the title and show the plot
    ax.set_title(f"Building Damage Pattern in {city_name}")
    ax.legend

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
   


