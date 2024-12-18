import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class RecordLinking:
    def __init__(self, company_data, entity_dict, model='sentence-transformers/all-MiniLM-L6-v2'):
        self.semantic_model = SentenceTransformer(model)
        self.df_company = company_data
        self.index = None 
        self.df_company = self._embed_company_data()
        self.entity_dict = entity_dict
        self.entity_dict_output = self._map_entities_to_similar()
        s

    def _embed_company_data(self):
        df = self.df_company
        semantic_model = self.semantic_model 
        df['Embedding'] = df['short name'].apply(lambda x: semantic_model.encode(x))

        # Build FAISS index
        embedding_matrix = np.vstack(df['Embedding'].to_numpy())
        self.index = faiss.IndexFlatL2(embedding_matrix.shape[1])
        self.index.add(embedding_matrix)
        self.company_data = df
        return df


    def find_similar(self, entity_value, top_k=1):
        df = self.company_data
        semantic_model = self.semantic_model 
        entity_embedding = semantic_model.encode(entity_value)
        distances, indices = self.index.search(np.array([entity_embedding]), top_k)
        return df.iloc[indices[0]].assign(Distance=distances[0])

    def _map_entities_to_similar(self):
        updated_dict = {} 

        for key, value in self.entity_dict.items():
            if isinstance(value, list):
                # If value is a list, find similar matches for each item
                updated_dict[key] = [self.find_similar(v, top_k=1)['short name'].iloc[0] for v in value]
            else:
                # If value is a string, find the most similar match
                updated_dict[key] = self.find_similar(value, top_k=1)['short name'].iloc[0]

        # self.entity_dict_output = updated_dict
        return updated_dict


    
        
















"""
def our_new_func()
ARGS:
    dictionary: ids: values: ['GME', 'GAMESTOP'] .... and so on
    COMPANY DATA: Columns: 'ticker', 'short name'

OUTPUT:
    Updated dictionary: ids: ... values: ['GME', GME'] ...

"""
