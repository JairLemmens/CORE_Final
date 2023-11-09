# Imports for Openstreetmap
import os 
# Imports for Geopandas
from shapely.geometry import Polygon
# Imports for statistical analysis
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

Own_filepath = r"C:\Users\Véronique\OneDrive\Documenten\Master bouwkunde\Leerjaar 4\CORE"
# Create a Back-End folder with the following folders in there:
# Area excel files
# Building properties
    # Islahiye
        # Statistical Analysis
    # Nurdagi
        # Statistical Analysis
    # Marash
        # Statistical Analysis
    # Turkoglu
        # Statistical Analysis
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

# Import the created dataframes from the Main_Dataframe'
# It is possible to make the statistical analysis directly, however since the buildig properties data is randomnized it is recommendable to save it first as an excel or CSV
df_islahiye = pd.read_excel(Own_filepath + r"\09_Back-End\Dataframes\Islahiye\Df_Islahiye.xlsx")
df_nurdagi = pd.read_excel(Own_filepath + r"\09_Back-End\Dataframes\Nurdagi\Df_Nurdagi.xlsx")
df_marash = pd.read_excel(Own_filepath + r"\09_Back-End\Dataframes\Marash\Df_Marash.xlsx")
df_turkoglu = pd.read_excel(Own_filepath + r"\09_Back-End\Dataframes\Turkoglu\Df_Turkoglu.xlsx")

# The column construction year contains invalid data this needs to be removed
def preprocess_construction_year(df):
    """
    Explanation: 
    Preprocess the 'Construction Year' column in the DataFrame.

    Parameters:
    df (DataFrame): The DataFrame containing the data.

    Returns:
    DataFrame: The preprocessed DataFrame.
    """
    # Make a copy of the DataFrame to avoid SettingWithCopyWarning
    df = df.copy()

    # Remove rows with 'Unknown' values
    df = df[df['Construction Year'] != 'Unknown']

    # Calculate the median year for imputation
    median_year = df['Construction Year'].median()

    # Replace 'Unknown' with the median year
    df['Construction Year'].replace('Unknown', median_year, inplace=True)

    # Convert the column to integers, encoding 'Unknown' as -1
    df['Construction Year'] = df['Construction Year'].apply(lambda x: -1 if x == 'Unknown' else int(x))

    return df

# New dataframes with valid data
df_islahiye_2 = preprocess_construction_year(df_islahiye)
df_nurdagi_2 = preprocess_construction_year(df_nurdagi)
df_marash_2 = preprocess_construction_year(df_marash)
df_turkoglu_2 = preprocess_construction_year(df_turkoglu)

# Columns that can be investigated on their coherencies 
columns_for_heatmap = ['Damage Level', 'Longitude', 'Latitude', 'Area (m²)', 'Number of Stories', 'Occupants', 'Construction Year']

# Corelation between all the parameters to see which parameters effect the Damage Level
def create_correlation_heatmap(df, columns_for_heatmap, name="some_city", save=False, filepath=None, show=True, close=True):
    """
    Explanation: 
    Create a correlation heatmap for selected columns in the DataFrame.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - columns_for_heatmap (list): List of column names for the heatmap.
    - name (str): Name of the city or location.
    - save (bool): Whether to save the plot as an image.
    - filepath (str): Filepath to save the image.
    - show (bool): Whether to display the plot.
    - close (bool): Whether to close the plot after displaying or saving.
    """
    
    # Create a correlation matrix
    correlation_matrix = df[columns_for_heatmap].corr()

    # Create the heatmap with custom text orientation
    plt.figure(figsize=(10, 10))
    heatmap = sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, 
                     xticklabels=True, yticklabels=True)
    
    # Rotate x-axis labels (columns) to be horizontal
    plt.xticks(rotation=0)   
    plt.title(f'Heatmap_{name}')  

    if save and filepath:
        # Ensure the directory exists or create it
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save the plot as an image
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    if show:
        plt.show()
    if close:
        plt.close()

# Heatmaps using all the columns to see if there is a corelation between certain columns
create_correlation_heatmap(df_islahiye_2, columns_for_heatmap, name="Islahiye", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Islahiye\Heatmap.jpeg", show=False, close=True)
create_correlation_heatmap(df_nurdagi_2, columns_for_heatmap, name="Nurdagi", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Nurdagi\Heatmap.jpeg", show=False, close=True)
create_correlation_heatmap(df_marash_2, columns_for_heatmap, name="Marash", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Marash\Heatmap.jpeg", show=False, close=True)
create_correlation_heatmap(df_turkoglu_2, columns_for_heatmap, name="Turkoglu", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Turkoglu\Heatmap.jpeg", show=False, close=True)                            

# Check the division of the damage levels for the cities in Turkey
def create_damage_level_pie_chart(df, name="some_city", save=False, filepath=None, show=True, close=True):
    """
    Explanation: 
    Create a pie chart for the distribution of damage levels.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - name (str): Name of the city or location.
    - save (bool): Whether to save the plot as an image.
    - filepath (str): Filepath to save the image.
    - show (bool): Whether to display the plot.
    - close (bool): Whether to close the plot after displaying or saving.
    """
    # Filter the DataFrame to include only damage levels between 2 and 4 (inclusive)
    filtered_df = df[df['Damage Level'].between(2, 4)]

    # Count the occurrences of damage levels within the specified range
    data = filtered_df['Damage Level'].value_counts()

    # Create a pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(data, labels=data.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Damage Level Distribution (2-4)_{name}')

    if show:
        plt.show()
    if save and filepath:
        # Ensure the directory exists or create it
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save the plot as an image
        plt.savefig(filepath, dpi=300, bbox_inches='tight')  
    if close:
        plt.close()

# Damage level distribution
create_damage_level_pie_chart(df_islahiye_2, name="Islahiye", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Islahiye\Damage distribition.jpeg", show=False, close=True)
create_damage_level_pie_chart(df_nurdagi_2, name="Nurdagi", save=True, filepath=Own_filepath + r"\CORE\09_Back-End\Dataframes\Nurdagi\Damage distribition.jpeg", show=False, close=True)
create_damage_level_pie_chart(df_marash_2, name="Marash", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Marash\Damage distribition.jpeg", show=False, close=True)
create_damage_level_pie_chart(df_turkoglu_2, name="Turkoglu", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Turkoglu\Damage distribition.jpeg", show=False, close=True)

# Box plot to compare the distribution of the number of stories for each damage level
def create_damage_level_box_plot(df, name="some_city", save=False, filepath=None, show=True, close=True):
    """
    Explanation: 
    Create a box plot to compare the distribution of the number of stories for each damage level.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - name (str): Name of the city or location.
    - save (bool): Whether to save the plot as an image.
    - filepath (str): Filepath to save the image.
    - show (bool): Whether to display the plot.
    - close (bool): Whether to close the plot after displaying or saving.
    """
    
    # Create a box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Damage Level', y='Number of Stories', hue='Damage Level', data=df, palette='Set3', legend=False)
    plt.title(f'Number of Stories by Damage Level_{name}')

    if show:
        plt.show()
    if save and filepath:
        # Ensure the directory exists or create it
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save the plot as an image
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    if close:
        plt.close()

# Boxplots to see if the number of stories has an impact on the damage level.
create_damage_level_box_plot(df_islahiye_2, name="Islahiye", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Islahiye\Damage compared with number of stories.jpeg", show=False, close=True) 
create_damage_level_box_plot(df_nurdagi_2, name="Nurdagi", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Nurdagi\Damage compared with number of stories.jpeg", show=False, close=True) 
create_damage_level_box_plot(df_marash_2, name="Marash", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Marash\Damage compared with number of stories.jpeg", show=False, close=True) 
create_damage_level_box_plot(df_turkoglu_2, name="Turkoglu", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Turkoglu\Damage compared with number of stories.jpeg", show=False, close=True) 

# Box Plot: Damage Level vs. Construction Year
def create_damage_vs_construction_year_box_plot(df, name="some_city", save=False, filepath=None, show=True, close=True):
    """
    Explanation: 
    Create a box plot to compare damage level vs. construction year.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - name (str): Name of the city or location.
    - save (bool): Whether to save the plot as an image.
    - filepath (str): Filepath to save the image.
    - show (bool): Whether to display the plot.
    - close (bool): Whether to close the plot after displaying or saving.
    """
    
    # Create a box plot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Damage Level', y='Construction Year', hue='Damage Level', data=df, palette='Set3', legend=False)
    plt.title(f'Damage Level vs. Construction Year_{name}')

    if show:
        plt.show()
    if save and filepath:
        # Ensure the directory exists or create it
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save the plot as an image
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    if close:
        plt.close()

# Bowxplots per city damage level compared with the construction year, with the goal to see if the construction year has an infuence on the damage level.
create_damage_vs_construction_year_box_plot(df_islahiye_2, name="Islahiye", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Islahiye\Damage compared with construction year.jpeg", show=False, close=True) 
create_damage_vs_construction_year_box_plot(df_nurdagi_2, name="Nurdagi", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Nurdagi\Damage compared with construction year.jpeg", show=False, close=True) 
create_damage_vs_construction_year_box_plot(df_marash_2, name="Marash", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Marash\Damage compared with construction year.jpeg", show=False, close=True) 
create_damage_vs_construction_year_box_plot(df_turkoglu_2, name="Turkoglu", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Turkoglu\Damage compared with construction year.jpeg", show=False, close=True) 

# Create histograms for attributes like "Area (m²)," "Occupants," and more, grouped by damage level. This allows you to see how these attributes are distributed within each damage category.
def create_histograms_by_damage_level(df, location, name="some_city", save=False, filepath=None, show=True, close=True):
    """
    Explanation: 
    Create histograms for the selected location coordinate (Latitude or Longitude) grouped by damage level.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - location (str): Either 'Latitude' or 'Longitude' to specify which location coordinate to use.
    - name (str): Name of the city or location.
    - save (bool): Whether to save the plot as an image.
    - filepath (str): Filepath to save the image.
    - show (bool): Whether to display the plot.
    - close (bool): Whether to close the plot after displaying or saving.
    """
    plt.figure(figsize=(10, 6))
    
    if location == 'Latitude':
        data_col = df[df['Damage Level'] > 1]['Latitude']
    elif location == 'Longitude':
        data_col = df[df['Damage Level'] > 1]['Longitude']
    else:
        raise ValueError("Location must be 'Latitude' or 'Longitude'.")

    sns.histplot(data_col, element="step", common_norm=False, bins=30)
    plt.title(f'Histograms for {location} by Damage Level_{name}')

    if show:
        plt.show()
    if save and filepath:
        # Ensure the directory exists or create it
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save the plot as an image
        plt.savefig(filepath, dpi=300, bbox_inches='tight')  
    if close:
        plt.close()
    

# What is the infleunce of the location compared with the damage grade.
create_histograms_by_damage_level(df_islahiye_2, 'Latitude', name="Islahiye", save=True, filepath=Own_filepath + r"\CORE\09_Back-End\Dataframes\Islahiye\Damage compared with location Latitude.jpeg", show=False, close=True)
create_histograms_by_damage_level(df_nurdagi_2, 'Latitude', name="Nurdagi", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Nurdagi\Damage compared with location Latitude.jpeg", show=False, close=True) 
create_histograms_by_damage_level(df_marash_2, 'Latitude', name="Marash", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Marash\Damage compared with location Latitude.jpeg", show=False, close=True) 
create_histograms_by_damage_level(df_turkoglu_2, 'Latitude', name="Turkoglu", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Turkoglu\Damage compared with location Latitude.jpeg", show=False, close=True) 
create_histograms_by_damage_level(df_islahiye_2, 'Longitude', name="Islahiye", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Islahiye\Damage compared with location Longitude.jpeg", show=False, close=True) 
create_histograms_by_damage_level(df_nurdagi_2, 'Longitude', name="Nurdagi", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Nurdagi\Damage compared with location Longitude.jpeg", show=False, close=True) 
create_histograms_by_damage_level(df_marash_2, 'Longitude', name="Marash", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Marash\Damage compared with location Longitude.jpeg", show=False, close=True) 
create_histograms_by_damage_level(df_turkoglu_2, 'Longitude', name="Turkoglu", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Turkoglu\Damage compared with location Longitude.jpeg", show=False, close=True) 

# Stacked Bar Chart: Damage Level vs. Construction Type
def create_damage_vs_construction_type_stacked_bar_chart(df, name="some_city", save=False, filepath=None, show=True, close=True):
    """
    Explanation: 
    Create a stacked bar chart to visualize the distribution of damage levels vs. construction types.

    Parameters: 
    - df (DataFrame): The DataFrame containing the data.
    - name (str): Name of the city or location.
    - save (bool): Whether to save the plot as an image.
    - filepath (str): Filepath to save the image.
    - show (bool): Whether to display the plot.
    - close (bool): Whether to close the plot after displaying or saving.
    """   
    # Create a stacked bar chart
    plt.figure(figsize=(10, 6))
    crosstab = pd.crosstab(df['Damage Level'], df['Construction Type'])
    crosstab.div(crosstab.sum(1), axis=0).plot(kind='bar', stacked=True, colormap='viridis')
    plt.title(f'Damage Level vs. Construction Type_{name}')

    if show:
        plt.show()
    if save and filepath:
        # Ensure the directory exists or create it
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save the plot as an image
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    if close:
        plt.close()

# Bowxplots per city damage level compared with the construction type, with the goal to see if the construction type has an infuence on the damage level.
create_damage_vs_construction_type_stacked_bar_chart(df_islahiye_2, name="Islahiye", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Islahiye\Damage compared with construction type.jpeg", show=False, close=True) 
create_damage_vs_construction_type_stacked_bar_chart(df_nurdagi_2, name="Nurdagi", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Nurdagi\Damage compared with construction type.jpeg", show=False, close=True) 
create_damage_vs_construction_type_stacked_bar_chart(df_marash_2, name="Marash", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Marash\Damage compared with construction type.jpeg", show=False, close=True) 
create_damage_vs_construction_type_stacked_bar_chart(df_turkoglu_2, name="Turkoglu", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Turkoglu\Damage compared with construction type.jpeg", show=False, close=True) 

#####
def create_damage_vs_construction_type_stacked_bar_chart_specified(df, name="some_city", save=False, filepath=None, show=True, close=True):
    """
    Explanation: 
    Create a stacked bar chart to visualize the distribution of damage levels between 3 and 4 vs. construction types and calculate the relative occurrence of these damage levels within each construction type.

    Parameters: 
    - df (DataFrame): The DataFrame containing the data.
    - name (str): Name of the city or location.
    - save (bool): Whether to save the plot as an image.
    - filepath (str): Filepath to save the image.
    - show (bool): Whether to display the plot.
    - close (bool): Whether to close the plot after displaying or saving.
    """
    # Filter the data for damage levels between 3 and 4
    filtered_df = df[(df['Damage Level'] >= 3) & (df['Damage Level'] <= 4)]

    # Create a stacked bar chart
    plt.figure(figsize=(10, 6))
    crosstab = pd.crosstab(filtered_df['Damage Level'], filtered_df['Construction Type'])
    crosstab.div(crosstab.sum(1), axis=0).plot(kind='bar', stacked=True, colormap='viridis')
    plt.title(f'Damage Levels between 3 and 4 vs. Construction Type_{name}')

    # Calculate the relative occurrence of damage levels between 3 and 4 within construction types
    damage_level_counts = filtered_df.groupby(['Construction Type', 'Damage Level'])['Damage Level'].count()
    total_damage_level_counts = filtered_df['Damage Level'].value_counts()
    relative_occurrence = (damage_level_counts / total_damage_level_counts).unstack().fillna(0)

    # Visualize the relative occurrence
    plt.figure(figsize=(10, 6))
    relative_occurrence.plot(kind='bar', stacked=True, colormap='viridis')
    plt.title(f'Relative Occurrence of Damage Levels between 3 and 4 within Construction Types_{name}')

    if show:
        plt.show()

    if save and filepath:
        # Ensure the directory exists or create it
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save the plot as an image
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

# Bowxplots per city damage level compared with the construction type, with the goal to see if the construction type has an infuence on the damage level.
create_damage_vs_construction_type_stacked_bar_chart_specified(df_islahiye_2, name="Islahiye", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Islahiye\Damage compared with construction type, specified.jpeg", show=True, close=True) 
create_damage_vs_construction_type_stacked_bar_chart_specified(df_nurdagi_2, name="Nurdagi", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Nurdagi\Damage compared with construction type,specified.jpeg", show=True, close=True) 
create_damage_vs_construction_type_stacked_bar_chart_specified(df_marash_2, name="Marash", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Marash\Damage compared with construction type,specified.jpeg", show=True, close=True) 
create_damage_vs_construction_type_stacked_bar_chart_specified(df_turkoglu_2, name="Turkoglu", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Turkoglu\Damage compared with construction,specified type.jpeg", show=True, close=True) 





# What is the influence of the location
def create_location_damage_scatter_plot(df, name="some_city", save=False, filepath=None, show=True, close=True):
    """
    Discription
    Create a scatter plot of location (latitude and longitude) with damage level.

    Parameters:
    - df (DataFrame): The DataFrame containing the data.
    - name (str): Name of the city or location (default is "some_city").
    - save (bool): Whether to save the plot as an image (default is False).
    - filepath (str): The file path to save the plot image (required if save is True).
    - show (bool): Whether to display the plot (default is True).
    - close (bool): Whether to close the plot after displaying or saving (default is True).

    """
    filtered_df = df[df['Damage Level'].between(2, 4)]

    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(filtered_df['Longitude'], filtered_df['Latitude'], c=filtered_df['Damage Level'], cmap='coolwarm', s=50)
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title(f'Scatter Plot of Location vs. Damage Level_{name}')

    # Adding a colorbar for the damage level
    colorbar = plt.colorbar(scatter)
    colorbar.set_label('Damage Level')

    if show:
        plt.show()
    if save and filepath:
        # Ensure the directory exists or create it
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # Save the plot as an image
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
    if close:
        plt.close()
    
create_location_damage_scatter_plot(df_islahiye_2, name="Islahiye", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Islahiye\Damage compared with location scatter.jpeg", show=False, close=True) 
create_location_damage_scatter_plot(df_nurdagi_2, name="Nurdagi", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Nurdagi\Damage compared with location scatter.jpeg", show=False, close=True) 
create_location_damage_scatter_plot(df_marash_2, name="Marash", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Marash\Damage compared with location scatter.jpeg", show=False, close=True) 
create_location_damage_scatter_plot(df_turkoglu_2, name="Turkoglu", save=True, filepath= Own_filepath + r"\CORE\09_Back-End\Dataframes\Turkoglu\Damage compared with location scatter.jpeg", show=False, close=True) 



