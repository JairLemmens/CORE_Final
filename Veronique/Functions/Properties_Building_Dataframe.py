import random

# Generate building data using the provided function
def gen_buildingdata_gaziantep(count):
    """
    Explanation: 
    Randomnizer of the following properties: construction type, year of construction and amount of stories
        
    City        Province
    Islahiye    Gaziantep
    Nurdagi     Gaziantep

    Parameters: 
    Count (amount of buildings)

    Returns: 
    Building data gaziantep
    """
    building_data_gaziantep = []

    for i in range(1, count + 1):
        building_name = f'Building {i}'

        # Randomize number of stories based on given percentages
        stories_prob = random.random()
        if stories_prob < 0.327:  # 32.7%
            num_stories = random.randint(1, 2)
        elif stories_prob < 0.637:  # 31%
            num_stories = random.randint(3, 6)
        else:  # 36.3%
            num_stories = random.randint(7, 10)

        # Randomize construction year based on given percentages
        construction_year_prob = random.random()
        if construction_year_prob < 0.066:  # 6.6%
            construction_year = random.randint(1800, 1979)  # Prior to 1980
        elif construction_year_prob < 0.325:  # 25.9%
            construction_year = random.randint(1980, 2000)  # 1981 to 2000
        elif construction_year_prob < 0.741:  # 51.6%
            construction_year = random.randint(2001, 2023)  # 2001 to 2023
        else:  # 15.9%
            construction_year = 'Unknown'

        # Randomize construction type based on given percentages
        construction_type_prob = random.random()
        if construction_type_prob < 0.867:  # 86.7%
            construction_type = 'Reinforced Concrete'
        elif construction_type_prob < 0.891 + 0.024:  # 2.4%
            construction_type = 'Steel'
        elif construction_type_prob < 0.926 + 0.035:  # 3.5%
            construction_type = 'Masonry'
        elif construction_type_prob < 0.962 + 0.036:  # 3.6%
            construction_type = 'Prefabricated'
        else:  # 3.8%
            construction_type = 'Others'

        building_data_gaziantep.append({
            'Building Name': building_name,
            'Number of Stories': num_stories,
            'Construction Year': construction_year,
            'Construction Type': construction_type
        })
    return building_data_gaziantep
  
# Generate building data using the provided function
def gen_buildingdata_kahramanmaraş(count):
    """
    Explanation: 
    Randomnizer of the following properties: construction type, year of construction and amount of stories

    City        Province
    Marash      Kahramanmaraş
    Turkoglu    Kahramanmaraş
    
    Parameters: 
    count (amount of buildings)

    Returns: 
    building data Kahramanmaraş
    """

    building_data_kahramanmaraş = []

    for i in range(1, count + 1):
        building_name = f'Building {i}'

        # Randomize number of stories based on given percentages
        stories_prob = random.random()
        if stories_prob < 0.5:  # 50%
            num_stories = random.randint(1, 2)
        elif stories_prob < 0.717:  # 21.7% 
            num_stories = random.randint(3, 6)
        else:  # 28.3%
            num_stories = random.randint(7, 10)

        # Randomize construction year based on given percentages
        construction_year_prob = random.random()
        if construction_year_prob < 0.117:  # 11.7%
            construction_year = random.randint(1800, 1979)  # Prior to 1980
        elif construction_year_prob < 0.386:  # 26.9% 
            construction_year = random.randint(1980, 2000)  # 1981 to 2000
        elif construction_year_prob < 0.967:  # 58.1% 
            construction_year = random.randint(2001, 2023)  # 2001 to 2023
        else:  # 3.3%
            construction_year = 'Unknown'

        # Randomize construction type based on given percentages
        construction_type_prob = random.random()
        if construction_type_prob < 0.867:  # 86.7%
            construction_type = 'Reinforced Concrete'
        elif construction_type_prob < 0.891 + 0.024:  # 2.4%
            construction_type = 'Steel'
        elif construction_type_prob < 0.926 + 0.035:  # 3.5%
            construction_type = 'Masonry'
        elif construction_type_prob < 0.962 + 0.036:  # 3.6%
            construction_type = 'Prefabricated'
        else:  # 3.8%
            construction_type = 'Others'

        building_data_kahramanmaraş.append({
            'Building Name': building_name,
            'Number of Stories': num_stories,
            'Construction Year': construction_year,
            'Construction Type': construction_type
        })
    return building_data_kahramanmaraş

