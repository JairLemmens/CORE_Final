import matplotlib.pyplot as plt
from shapely.geometry import box

def gen_map_damage_buildings(collapsed_gpkg, name="some_city", save=False, filepath=None, xlim=None, ylim=None, show=False, close=True):
    """
    Explanation: 
    Plot building damage pattern from 0 to 4.

    Parameters:
    building_gdf (geopandas.GeoDataFrame): GeoDataFrame containing building shapes and damage information.
    city_name (str): Name of the city for which to plot the damage pattern.

    Returns:
    Matplotlib plot showing the damage pattern.
    """
    
    # Assuming you have the 'collapsed' column in the GeoDataFrame
    mean_collapsed = collapsed_gpkg['dmg'].mean()

    # Plot the buildings with color representing the mean 'collapsed' value
    fig, ax = plt.subplots(figsize=(15, 15))
    collapsed_gpkg.plot(column='dmg', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)

    # Add a legend
    legend_title = f'Collapsed Buildings_{name}'
    legend_labels = [f'{mean_collapsed:.2f}']
          
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

     

    


