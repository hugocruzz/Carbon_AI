# %%
import openai
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from openai.embeddings_utils import get_embedding

# %%
with open(r"C:\Users\cruz" + r'\API_openAI.txt', 'r') as f:
    read_api_key = f.readline()
    openai.api_key = read_api_key
os.environ["OPENAI_API_KEY"] = read_api_key

# Embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # Encoding for text-embedding-ada-002
max_tokens = 8000  # Max tokens for ada-002

embed_source = False

if embed_source:
    path_source= r"data\achats_EPFL/Liste produits CATALYSE SV 2022.xlsx"
    input_column = ["Libellé FR Famille", "Libellé FR Sous-sous-famille"]
else: 
    path_source = "data/achats_EPFL/achats_100_embedded.pkl"

output_path = r"data\Achats_CO2_Famille1.xlsx"
database_path =r"data\unspsc_ada_embeddings.pkl"

def batch_embeddings(texts, batch_size=1500):
    total_batches = (len(texts) + batch_size - 1) // batch_size  # Calculate the total number of batches
    embeddings = []

    for i in range(0, len(texts), batch_size):
        current_batch = (i // batch_size) + 1  # Calculate the current batch number
        print(f"Processing batch {current_batch} of {total_batches}...")  # Print the progress
        batch = texts[i:i+batch_size]
        # Assuming get_embedding is a function that processes each batch
        # Replace the following line with your actual function call to get embeddings
        batch_embeddings = [get_embedding(text, engine=embedding_model) for text in batch]
        embeddings.extend(batch_embeddings)
    
    print("All batches processed.")
    return embeddings

def input_embedding(path_source, input_column, path_embedded=None):
    df_achats = pd.read_excel(path_source).head(100)
    df_achats.fillna("", inplace=True)
    df_achats['combined'] = df_achats[input_column].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
    combined_unique = df_achats['combined'].unique()
    if len(combined_unique) > 500:
        raise ValueError(f"This might take a long time to run. The number of unique values is {len(combined_unique)}. Please reduce the number of unique values.")
    
    embeddings_combine_unique = batch_embeddings(combined_unique.tolist(), batch_size=1000)
    df_combined_unique = pd.DataFrame(combined_unique)
    df_combined_unique["embedding"]  = embeddings_combine_unique
    df_achats = df_achats.merge(df_combined_unique, left_on="combined", right_on=0)
    df_achats.drop(0, axis=1, inplace=True)
    if path_embedded:
        df_achats.to_pickle(path_embedded)
    return df_achats

# %%
#Check if path_embedded exists
if not embed_source:
    df_source = pd.read_pickle(path_source)
else:
    df_source = input_embedding(path_source, input_column, path_source.replace(".xlsx", "_embedded.pkl"))

# %% Start calculating similarity matrix
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

# Combine all updates into a single DataFrame
updates_df = pd.concat(updates_list)

updates_df.to_excel(output_path, index=False)


max_similarity_score, closest_indices = mapping_embeddings(df_source, df_target)
df_source["closest_indices"] = closest_indices
df_source["max_similarity_score"] = max_similarity_score
#Merge df_source and df_target based on "closest_indices" and df_target.index
df_source = df_source.merge(df_target, left_on = "closest_indices", right_index = True, how = "left")
df_source["Flag"] = df_source["max_similarity_score"] < 0.8
df_source.to_excel(output_path, index=False)


#df_source["CO2eq_kg"] = df_source["Prix total"] * df_source["FEL1P5"]