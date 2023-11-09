#Imports for Geopandas
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle
from shapely.geometry import box

def plot_building_damage_pattern_visualisation(building_gdf, city_name, save=False, filepath=None, xlim=None, ylim=None, show=False, close=True):
    """
    Explanation:
    Plot building damage pattern with circles of varying sizes based on damage grade.

    Parameters:
    building_gdf (geopandas.GeoDataFrame): GeoDataFrame containing building shapes and damage information.
    city_name (str): Name of the city for which to plot the damage pattern.

    Returns:
    Matplotlib plot showing the damage pattern with circles.
    """
    circle_patches = []
    circle_colors = {'undamaged': 'green', 'low_damage': 'yellow', 'high_damage': 'red'}

    for idx, row in building_gdf.iterrows():
        color = circle_colors['undamaged']
        if row['dmg'] > 0:
            if row['dmg'] <= 1.0:
                color = circle_colors['low_damage']
                circle_radius = 10
            else:
                color = circle_colors['high_damage']
                circle_radius = 20  # Adjust the radius for high damage

            if row['geometry'] and row['geometry'].is_valid:
                centroid = row['geometry'].centroid
                circle = Circle((centroid.x, centroid.y), radius=circle_radius, color=color, alpha=0.5)
                circle_patches.append(circle)

    circle_collection = PatchCollection(circle_patches, match_original=True)

    building_gdf.plot(column="dmg", cmap='OrRd', legend=True, legend_kwds={'label': "Damage levels"})
    ax = building_gdf.plot(figsize=(10, 10), alpha=0.5)
    plt.gca().add_collection(circle_collection)
    plt.title(f"Building Damage Pattern in {city_name}")

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if save and filepath:
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if close:
        plt.close()
