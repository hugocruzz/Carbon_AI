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

output_path = r"data\Achats_CO2.xlsx"
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

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Function to batch process embeddings
def batch_embeddings(texts, batch_size=1500):
    total_batches = (len(texts) + batch_size - 1) // batch_size
    embeddings = []

    for i in range(0, len(texts), batch_size):
        current_batch = (i // batch_size) + 1
        print(f"Processing batch {current_batch} of {total_batches}...")
        batch = texts[i:i+batch_size]
        # Assuming get_embedding is a function that processes each batch
        batch_embeddings = [get_embedding(text, engine=embedding_model) for text in batch]
        embeddings.extend(batch_embeddings)
    
    print("All batches processed.")
    return embeddings


# Load source and target data
if not embed_source:
    df_source = pd.read_pickle(path_source)
else:
    df_source = input_embedding(path_source, input_column, path_source.replace(".xlsx", "_embedded.pkl"))

df_target = pd.read_pickle(database_path)
df_target['Code'] = df_target['Code'].astype(str)
# Function to map source embeddings to target embeddings at a specific level
def map_embeddings_at_level(df_source, df_target):
    source_embedding = np.array(df_source['embedding'].tolist())
    target_embedding = np.array(df_target['embedding'].tolist())
    similarity_matrix = cosine_similarity(source_embedding, target_embedding)

    closest_indices = np.argmax(similarity_matrix, axis=1)
    max_similarity_score = np.max(similarity_matrix, axis=1)
    return max_similarity_score, closest_indices

df_final = df_source.copy()
# Iterate through each level of the hierarchy
for level in range(1, 5):
    if level == 1:
        pattern = r'\d{2}000000$'
    elif level == 2:
        pattern = r'\d{4}0000$'
    elif level == 3:
        pattern = r'\d{6}00$'
    else:  # level == 4
        pattern = r'\d{8}$'
    # Filter df_target for the current level
    df_target_level = df_target[df_target['Code'].str.contains(pattern)]

    # Map embeddings at the current level
    max_similarity_score, closest_indices = map_embeddings_at_level(df_source, df_target_level)
    df_final[f'closest_indice_level_{level}'] = closest_indices
    df_final[f'max_similarity_score_level_{level}'] = max_similarity_score
    df_final[f'Unspsc_level_{level}_code'] = df_target_level.iloc[closest_indices]['Code'].values
    df_final[f'USNSPSC_Level_{level}_english_name'] = df_target_level.iloc[closest_indices]['English Name'].values

df_final.to_excel(output_path, index=False)


