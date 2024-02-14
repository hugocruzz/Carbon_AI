# %%
import openai
import os
import pandas as pd
import tiktoken
import numpy as np
from ast import literal_eval
from sklearn.metrics.pairwise import cosine_similarity

from openai.embeddings_utils import get_embedding


# %%
with open(r"C:\Users\cruz" + r'\API_openAI.txt', 'r') as f:
    read_api_key = f.readline()
    openai.api_key = read_api_key
os.environ["OPENAI_API_KEY"] = read_api_key

# %%
output_path = r"data\Achats_CO2.xlsx"
path_source= r"data\Liste produits CATALYSE SV 2022.xlsx"
path_source = "Ordinateur de bureau"
input_column = ["Libellé FR Famille", "Libellé FR Sous-sous-famille"]

#Keep unchanged 
path_embedded="data/processed/ACHATS_SV_embedded.pkl"
path_database =r"data\processed\nacres\NACRES_with_embeddings.csv"

def input_embedding(path_source, input_column, path_embedded=None):
    df_achats = pd.read_excel(path_source)
    df_achats.fillna("", inplace=True)
    df_achats['combined'] = df_achats[input_column].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    combined_unique = df_achats['combined'].unique()
    if len(combined_unique) > 500:
        raise ValueError(f"This might take a long time to run. The number of unique values is {len(combined_unique)}. Please reduce the number of unique values.")
    embeddings_combine_unique = [get_embedding(x, engine="text-embedding-ada-002", embedding_encoding = "cl100k_base") for x in combined_unique]
    df_combined_unique = pd.DataFrame(combined_unique)
    df_combined_unique["embedding"]  = embeddings_combine_unique
    df_achats = df_achats.merge(df_combined_unique, left_on="combined", right_on=0)
    df_achats.drop(0, axis=1, inplace=True)
    if path_embedded:
        df_achats.to_pickle(path_embedded)
    return df_achats

# %%
#Check if path_embedded exists
if os.path.exists(path_embedded):
    df_source = pd.read_pickle(path_embedded)
else:
    df_source = input_embedding(path_source, input_column, path_embedded)

# %%
#Check if path_source containes a .xlsx file
df_target = pd.read_pickle(path_source)
def mapping_embeddings(df_source, df_target):
    # Assuming df_NACRES_ada['embedding'] and df_unspsc['embedding'] are lists of lists
    # Convert these lists of embeddings into NumPy arrays
    nacres_embeddings = np.array(df_source['embedding'].tolist())
    unspsc_embeddings = np.array(df_target['embedding'].tolist())

    # Compute cosine similarity between all pairs of NACRES and UNSPSC embeddings
    # The result is a matrix of shape (len(df_NACRES_ada), len(df_unspsc))
    # Each element [i, j] represents the cosine similarity between the i-th NACRES embedding and the j-th UNSPSC embedding
    similarity_matrix = cosine_similarity(nacres_embeddings, unspsc_embeddings)

    # Find the index of the closest UNSPSC embedding for each NACRES embedding
    closest_indices = np.argmax(similarity_matrix, axis=1)

    # Extract the closest distances (cosine similarities)
    closest_distances = np.max(similarity_matrix, axis=1)
    return closest_distances, closest_indices

closest_distances, closest_indices = mapping_embeddings(df_source, df_target)
df_source["closest_indices"] = closest_indices
df_source["closest_distances"] = (1-closest_distances)
#Merge df_source and df_target based on "closest_indices" and df_target.index
df_source = df_source.merge(df_target, left_on = "closest_indices", right_index = True, how = "left", rsuffix = "_NACRES")
df_source["CO2eq_kg"] = df_source["Prix total"] * df_source["FEL1P5"]
df_source["Flag"] = df_source["closest_distances"] < 0.8
df_source.to_excel(output_path, index=False)