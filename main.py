"""
file: main.py
"""
import warnings
warnings.simplefilter('ignore')
import time
import pandas

# LOADING DATAHANDLER CLASS 
import data_preprocessing
CLASS_NAME = 'DataHandler' 
def create_and_save_data_class(class_name):
    DataHandler = data_preprocessing.DataHandler(class_name)
    DataHandler.saveDataHandlerClass(class_name)
create_and_save_data_class(CLASS_NAME)
DATAHANDLER = data_preprocessing.loadDataHandler(CLASS_NAME)


"""
#####################################
#####################################
#####################################
Functions for handling NER Pipeline
#####################################
#####################################
#####################################
"""


def filter_org_entities(entity_list):
    # Check if the value is a list; if not, return an empty list
    if not isinstance(entity_list, list):
        return []
    # Filter entities where 'entity' == 'ORG'
    return [entity for entity in entity_list if entity.get('entity') == 'ORG']


def print_column_types(df, column_name):
    print("Types in column:", column_name)
    print(df[column_name].apply(type))




def combine_org_entities_with_ids(df, column_name='entities'):
    """
    Combine 'ORG' words from the specified column into a dictionary where:
    - Key: Row index (ID) in the DataFrame.
    - Value: List of all 'ORG' words in that row.
    
    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - column_name (str): Name of the column containing lists of dictionaries.

    Returns:
    - dict: A dictionary with row IDs as keys and 'ORG' words as values.
    """
    combined_dict = {}

    for index, row in df.iterrows():
        entities = row[column_name]
        if isinstance(entities, list):  # Ensure it's a list
            org_words = [entity['word'] for entity in entities 
                         if isinstance(entity, dict) and entity.get('entity') == 'ORG']
            if org_words:  # Add only if ORG words are found
                combined_dict[index] = org_words

    return combined_dict

    

if __name__ == "__main__":
    start_time = time.time()
    class_name = 'DataHandler_class'

    
    ###########################
    #### Pipeline for NER #####
    ###########################

    ###### Remove unecessary entities #####
    df = DATAHANDLER.reddit_data
    result_dict = combine_org_entities_with_ids(df, 'entities')

    ###############################
    #### Pipeline for Mapping #####
    ###############################

    ##### Now we need to use the result dict into mapping function.
    ##### Then put it back inside the dataframe














    #print(df.columns)
    # df['filtered_entities'] = df['entities'].apply(filter_org_entities)
    # print(df['filtered_entities'].head())




   

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Total execution time: {execution_time} seconds")
 
