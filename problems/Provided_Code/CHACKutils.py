import numpy as np
import pandas as pd

def dataloader(multiIndex=False):
    """
    Utility function to facilitate loading CHACK data. Returns two separate dataframes
    
    Parameters:
    ----------
    multIndex (bool) - flag to set whether function returns species abundance data as multi-index dataframe or single-index dataframe. For project 1 start with False, for project 2 start with True
    
    Returns:
    -------
    abundancedata_species (DataFrame) - dataframe that contains the bacterial abundance data at the species level. Multi-index columns at all other taxonomic levels if multiIndex flag set to True
    output_metals (DataFrame) - dataframe with output values and metal concentrations at each location
    
    """
    
    abundancedata_species = pd.read_csv('CHACK2022_abundanceData.csv', header = [0,1,2,3,4,5,6,7], index_col=0)
    output_metals = pd.read_csv('CHACK2022_OutputsMetals.csv', index_col=0)
    #abundancedata_species = abundancedata_species.drop(columns = ['x location', 'y location', 'z location'], level=0)
    
    if not multiIndex:
        abundancedata_species = getTaxonomicLevelData('Species', abundancedata_species)
        abundancedata_species = abundancedata_species.rename(columns = {'Unnamed: 1_level_6':'x location', 'Unnamed: 2_level_6':'y location', 'Unnamed: 3_level_6':'z location'})
    
    #preppedData = pd.concat([output_metals, taxlevelAbundance], axis = 1)
    
    return abundancedata_species, output_metals

def getTaxonomicLevelData(TaxLevel, abundancedata_species):
    """
    Function to re-format species-level abundance data at the taxonomic level of the users choice. Turns a multi-index dataframe of data at the species level into a single-index dataframe at the level of users choice
    
    Background: Bacteria are classified at various taxonomic levels. As the specificity of the level increases, the number of bacteria at that level decreases until
    you reach the species level, which uniquely identifies a bacteria. You were provided bacterial abundance data at the species level, which tells you what fraction/percentage of the 
    total bacteria present at each location is made up of each bacterial species. It may be useful to consider a less specific taxonomic level in your analysis to reduce the amount of data you are working
    with. This function returns a new dataframe that contains abundance of each member of a taxonomic level of your choosing, by adding up all of the abundances of the individual species 
    that belong to that classification. 
    
    Parameters:
    ----------
    TaxLevel (string) - The taxonomic level you would like data at. Options are: Kingdom, Phylum, Class, Order, Family, Genus, Species. Spelling and capitaliztion count
    abundancedata_species (dataframe) - the abundance data dataframe returned from the CHACK dataloader function using the multiIndex=True flag. Must be in the same format, do not modify before passing to this function or else things will break. 
    
    Returns:
    -------
    newdf (dataframe) - A new dataframe that has members of the taxonomic level selected as column headers, with the abundance of each class at each location as row entries. Rows should add to 100
    
    """
    # Check that multi-index dataframe is passed: Use column labels being a tuple as a proxy for this
    
    assert type(abundancedata_species.columns[0]) == tuple, 'Dataframe passed to function is not multi-indexed. This function only takes the multi-indexed version of the species data. Check that you are loading the data using the CHACKutils function called "dataloader" with the optional argument "multiIndex = True" '
    
    # first cut location data away
    location_data = abundancedata_species[abundancedata_species.columns[:3]]
    location_data.columns = ['x location', 'y location', 'z location']
    species_data = abundancedata_species[abundancedata_species.columns[3:]]
    columns = species_data.columns
    
    # now we must distinguish unclassified bacteria at any level
    new_columns = []
    for column in columns:
        new_column = []
        for level in column:
            if not (level.startswith('Unclassified') or level.startswith('nan')):
                last_known_level = level
                new_column.append(level)
            else:
                new_column.append(last_known_level+'-Unclassified')
        new_columns.append(new_column)

    # convert new labels into index and assign to dataframe
    new_index = pd.MultiIndex.from_arrays(np.array(new_columns).T, names=columns.names)
    species_data.columns = new_index
    
    # aggregate and at the desired level
    taxleveltoIndex = {'Kingdom':0, 'Phylum':1, 'Class':2, 'Order':3, 'Family':4, 'Genus':5, 'Species':6}
    aggregated_data = species_data.sum(axis=1, level = taxleveltoIndex[TaxLevel], skipna=False)
    
    # concat the location data back
    aggregated_data = pd.concat([location_data, aggregated_data], axis=1)
    return aggregated_data