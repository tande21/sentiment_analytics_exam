"""
file name: NER_output_processing
purpose: Functions to process the output from NER

"""
from semantic_search_faiss import SemanticSearch
from record_linking import RecordLinking


# NER Processing 
def filter_org_entities(entity_list):
    if not isinstance(entity_list, list):
        return []
    return [entity for entity in entity_list if entity.get('entity') == 'ORG']


# NER Processing 
def combine_org_entities_with_ids(df, column_name='entities'):
    combined_dict = {}
    for index, row in df.iterrows():
        entities = row[column_name]
        if isinstance(entities, list):
            org_words = [entity['word'] for entity in entities 
                         if isinstance(entity, dict) and entity.get('entity') == 'ORG']
            if org_words:
                combined_dict[index] = org_words
    return combined_dict


# NER Processing 
def ensure_unique_values(my_dict):
    new_dict = {}
    for key, value in my_dict.items():
        if isinstance(value, list):
            seen = set()
            unique_values = []
            for item in value:
                if item == "Ticker not found":
                    if "Ticker not found" not in unique_values:
                        unique_values.append(item)
                elif item not in seen:
                    seen.add(item)
                    unique_values.append(item)
            new_dict[key] = unique_values
        else:
            new_dict[key] = value 
    return new_dict


# NER Processing
def remove_values_with_prefix(input_dict, prefix="##"):
    cleaned_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, list):
            cleaned_values = [v for v in value if not (isinstance(v, str) and v.startswith(prefix))]
            cleaned_dict[key] = cleaned_values
        else:
            cleaned_dict[key] = value
    return cleaned_dict
