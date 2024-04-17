# %%
import openai
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

with open(r"C:\Users\cruz" + r'\API_openAI.txt', 'r') as f:
    read_api_key = f.readline()
    openai.api_key = read_api_key
os.environ["OPENAI_API_KEY"] = read_api_key

# Embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # Encoding for text-embedding-ada-002
max_tokens = 8000  # Max tokens for ada-002


path_source = r"data\achats_EPFL\achats_SV_2022.pkl"
output_path = r"data\achats_SV_2022_NACRES_chatgpt.xlsx"
database_path =r"data\NACRES_embedded.pkl"

df_source = pd.read_pickle(path_source)
df_source.rename(columns={'embedding_y': 'embedding'}, inplace=True)

df_target = pd.read_pickle(database_path)
# Modified function to get top 10 matches
def mapping_embeddings(df_source, df_target, top_n=10):
    source_embedding = np.array(df_source['embedding'].tolist())
    target_embedding = np.array(df_target['embedding'].tolist())
    similarity_matrix = cosine_similarity(source_embedding, target_embedding)

    # Get indices of the top 10 closest matches
    closest_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_n]
    # Extract the top 10 similarity scores
    max_similarity_scores = np.sort(similarity_matrix, axis=1)[:, -top_n:][:, ::-1]
    return max_similarity_scores, closest_indices

# Calculate the top 10 matches
top_n = 10
similarity_scores, closest_indices = mapping_embeddings(df_source, df_target, top_n)

# Update df_source to include the top 10 matches
df_source["Similarity_scores"] = list(similarity_scores)
df_source["NACRES_names"] = [[df_target.loc[idx, "nacres.description.en"] for idx in row] for row in closest_indices]

def choose_best_match_batch(df, batch_size=5):
    # Set up API key
    openai.api_key = os.environ["OPENAI_API_KEY"] 
    for idx, row in df.iterrows():
            options = ", ".join([f'"{option}" ({index})' for index, option in enumerate(row['NACRES_names'])])
            prompt = f"Product Name: {row['Désignation article']}, Category: {row['Famille_libellé anglais']}, Provider: {row['Fournisseur']}. The product fits best in one of the categories: {options}. Based on the description, give the index of the best matched category."


            valid_response = False
            attempts = 0

            while not valid_response and attempts < 5:  # Limit the number of attempts to avoid infinite loops
                messages = [
                    {"role": "system", "content": "You are helping to match products to the most relevant categories by selecting an index. The output should not be anything else than an integer corresponding to the index."},
                    {"role": "user", "content": prompt}
                ]

                # Call the OpenAI Chat API for each product description
                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",  # Adjust the model name as needed
                    messages=messages
                )

                # Parse the response
                
                response_text = response.choices[0].message['content'].strip()
                try:
                    selected_index = int(response_text)
                    df.at[idx, 'Chosen_NACRES_index'] = selected_index
                    df.at[idx, 'Chosen_NACRES_name'] = row['NACRES_names'][selected_index]
                    valid_response = True
                except: 
                    try:
                        match = re.search(r'\((\d+)\)', response_text)
                        if match:
                            selected_index = int(match.group(1))
                        
                        df.at[idx, 'Chosen_NACRES_index'] = selected_index
                        df.at[idx, 'Chosen_NACRES_name'] = row['NACRES_names'][selected_index]
                        valid_response = True
                    except ValueError:
                        # If conversion to int fails, make another attempt
                        attempts += 1

            if not valid_response:
                df.at[idx, 'Chosen_NACRES_index'] = -1  # Indicates that no valid index was found
                df.at[idx, 'Chosen_NACRES_name'] = "Index not valid"

    return df

# Call the function with df_source
df_source = choose_best_match_batch(df_source, batch_size=5)
# Save updated DataFrame to Excel
df_source.to_excel("your_output_path.xlsx", index=False)
