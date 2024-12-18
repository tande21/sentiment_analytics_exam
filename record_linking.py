"""
def our_new_func()
ARGS:
    dictionary: ids: values: ['GME', 'GAMESTOP'] .... and so on
    COMPANY DATA: Columns: 'ticker', 'short name'

OUTPUT:
    Updated dictionary: ids: ... values: ['GME', GME'] ...

"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd


class RecordLinking:
    def __init__(self, company_data, entity_dict, model='sentence-transformers/all-MiniLM-L6-v2'):
        self.semantic_model = SentenceTransformer(model)
        self.df_company = company_data
        self.index = None 
        self.df_company = self._embed_company_data()
        self.entity_dict = entity_dict
        self.entity_dict_output = self._map_entities_to_similar()

    def _embed_company_data(self):
        df = self.df_company
        semantic_model = self.semantic_model 
        # df['Embedding'] = df['short name'].apply(lambda x: semantic_model.encode(x))

        df['combined'] = df['ticker'].astype(str) + ' ' + df['short name'].astype(str)
        df['Embedding'] = df['combined'].apply(lambda x: semantic_model.encode(x))

        # Build FAISS index
        embedding_matrix = np.vstack(df['Embedding'].to_numpy())
        self.index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        self.index.add(embedding_matrix)
        self.company_data = df
        return df


    def find_similar(self, entity_value, top_k=10):
        df = self.company_data
        semantic_model = self.semantic_model 
        entity_embedding = semantic_model.encode(entity_value)
        distances, indices = self.index.search(np.array([entity_embedding]), top_k)
        return df.iloc[indices[0]].assign(Distance=distances[0])

        #def _map_entities_to_similar(self):
        #    updated_dict = {} 

        #    for key, value in self.entity_dict.items():
        #        if isinstance(value, list):
        #            # If value is a list, find similar matches for each item
        #            updated_dict[key] = [self.find_similar(v, top_k=1)['short name'].iloc[0] for v in value]
        #        else:
        #            # If value is a string, find the most similar match
        #            updated_dict[key] = self.find_similar(value, top_k=1)['short name'].iloc[0]

        #    # self.entity_dict_output = updated_dict
        #    return updated_dict

        # def _map_entities_to_similar(self, keys_to_process=50):
        #     if keys_to_process is None:
        #         keys_to_process = []  # specify a list of keys you want to test
        #     
        #     updated_dict = {}
        #     for key, value in self.entity_dict.items():
        #         if key not in keys_to_process:
        #             continue  # skip keys not in keys_to_process
        # 
        #         if isinstance(value, list):
        #             updated_dict[key] = [self.find_similar(v, top_k=1)['short name'].iloc[0] for v in value]
        #         else:
        #             updated_dict[key] = self.find_similar(value, top_k=1)['short name'].iloc[0]
        # 
        #     return updated_dict
        # 

    def _map_entities_to_similar(self, limit=None, keys_to_process=None):
        updated_dict = {}
        items = list(self.entity_dict.items())
        
        # Apply limit to the number of items if specified
        if limit is not None:
            items = items[:limit]
        
        # Convert keys_to_process to a set for faster lookup if provided
        keys_to_process = set(keys_to_process) if keys_to_process else None
        
        for key, value in items:
            # Skip invalid keys (NaN or non-string types)
            #if pd.isna(key) or not isinstance(key, str):
            #    continue
            if pd.isna(key):
                continue

            
            # If keys_to_process is specified, skip unlisted keys
            if keys_to_process and key not in keys_to_process:
                continue
            
            if isinstance(value, list):
                # Skip invalid list values
                cleaned_values = [v for v in value if isinstance(v, str) and not pd.isna(v)]
                if not cleaned_values:
                    continue
                # updated_dict[key] = [self.find_similar(v, top_k=1)['short name'].iloc[0] for v in cleaned_values]
                updated_dict[key] = [self.find_similar(v, top_k=1)['ticker'].iloc[0] for v in cleaned_values]
            elif isinstance(value, str) and not pd.isna(value):
                # Process valid string values
                # updated_dict[key] = self.find_similar(value, top_k=1)['short name'].iloc[0]
                updated_dict[key] = self.find_similar(value, top_k=1)['ticker'].iloc[0]
            else:
                # Skip invalid or empty values
                continue
        
        return updated_dict
    
    
        

















