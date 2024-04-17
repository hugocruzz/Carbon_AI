# %%
import openai
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from openai.embeddings_utils import get_embedding

with open(r"C:\Users\cruz" + r'\API_openAI.txt', 'r') as f:
    read_api_key = f.readline()
    openai.api_key = read_api_key
os.environ["OPENAI_API_KEY"] = read_api_key

# Embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # Encoding for text-embedding-ada-002
max_tokens = 8000  # Max tokens for ada-002


path_source = r"data\achats_EPFL\achats_SV_2022_keywords_embeded.pkl"
output_path = r"data\achats_SV_2022_UNSPSC_keywords.xlsx"
database_path =r"data\unspsc_ada_embeddings.pkl"

df_source = pd.read_pickle(path_source)
df_source.rename(columns={'embedding_y': 'embedding'}, inplace=True)
df_target = pd.read_pickle(database_path)
df_target['Code'] = df_target['Code'].astype(str)

def mapping_embeddings(df_source, df_target):
    source_embedding = np.array(df_source['embedding'].tolist())
    target_embedding = np.array(df_target['embedding'].tolist())
    similarity_matrix = cosine_similarity(source_embedding, target_embedding)

    # Find the index of the closest UNSPSC embedding for each NACRES embedding
    closest_indices = np.argmax(similarity_matrix, axis=1)
    # Extract the closest distances (cosine similarities)
    max_similarity_score = np.max(similarity_matrix, axis=1)
    return max_similarity_score, closest_indices

updates_list = []  # List to collect updates for each Famille1 group
df_source['Famille1'] = df_source['Famille1'].apply(lambda x: str(int(x)) if pd.notnull(x) else None)

for famille1_value in df_source["Famille1"].unique():
    if pd.isna(famille1_value):
        df_target_filtered = df_target.copy()
        df_source_filtered = df_source[pd.isna(df_source["Famille1"])]
    else:
        sub_string_match = famille1_value.split("00")[0] + r'\d{6}'
        df_target_filtered = df_target[df_target["Code"].str.contains(f"{sub_string_match}", na=False)]
        df_source_filtered = df_source[df_source["Famille1"] == famille1_value]
        
        # Add original index as a column before resetting index
        df_target_filtered = df_target_filtered.reset_index().rename(columns={'index': 'original_index'})

    # Proceed with your embedding and similarity calculations
    max_similarity_score, closest_indices = mapping_embeddings(df_source_filtered, df_target_filtered)

    # Use 'original_index' to correctly reference rows in df_target_filtered
    df_source_filtered["UNSPSC_generated_max_similarity_score"] = max_similarity_score
    df_source_filtered["UNSPSC_generated_name"] = df_target_filtered.loc[closest_indices, "English Name"].values
    df_source_filtered["UNSPSC_generated_code"] = df_target_filtered.loc[closest_indices, "Code"].values
    updates_list.append(df_source_filtered)

updates_df = pd.concat(updates_list)
updates_df.to_excel(output_path, index=False)