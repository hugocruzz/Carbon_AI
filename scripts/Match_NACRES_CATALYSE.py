
import openai
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import json 

with open(r"C:\Users\cruz" + r'\API_openAI.txt', 'r') as f:
    read_api_key = f.readline()
    openai.api_key = read_api_key
os.environ["OPENAI_API_KEY"] = read_api_key

# Embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # Encoding for text-embedding-ada-002
max_tokens = 8000  # Max tokens for ada-002


path_source = r"data\Results\achats_GVG_lab_2022.pkl"
output_path = r"data\achats_GVG_NACRES_GPT.xlsx"
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
df_source.to_excel(r"data\achats_GVG_NACRES_embedding.xlsx", index=False)

def choose_best_match_GPT(df):
    #Convert df into a dictionnary:
    df.rename(columns={'Désignation article': 'Article name', 'Famille_libellé anglais':'Category', "Fournisseur": "Provider", "NACRES_names": "Options"}, inplace=True)
    df_dict = df[["Article name", "Category", "Provider", "Options"]].to_dict(orient='records')
    #Chunk df_dict into smaller chunks
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    example_json = {1:{"Choosen option": 'CLINICAL INVESTIGATION CONSUMABLES AND REAGENTS'}, 2:{"Choosen option": 'CELL BIOLOGY: ASSAY KITS, FUNCTIONAL ASSAYS - BIOCHEMICAL KITS'}}
    
    # Convert list of dictionaries into a dictionary with indices as keys
    df_dict = {i: row for i, row in enumerate(df_dict)}

    #df_dict is a dictionnary, split it into chunks of 10
    chunk_size = 7
    df_dict_chunks = [dict(list(df_dict.items())[i:i + chunk_size]) for i in range(0, len(df_dict), chunk_size)]


    for chunk in df_dict_chunks:
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            response_format={"type":"json_object"},
            messages=[
                {"role": "system", "content": """Provide output in valid JSON. Your are given a dictionnary in json format.
                For each elements, and based on the informations you have in the 'Article name', 'Category' and 'Provider' fields, 
                 you need to choose between 1 of the options in the 'Options' field.
                 The choosen option should be the most relevant to the 'Article name', 'Category' and 'Provider' fields.
                The data schema should be like this :""" + json.dumps(example_json)},
                {"role": "user", "content": json.dumps(chunk)}
            ]
        )
        data= chat_completion.choices[0].message.content
        data_json = json.loads(data)
        for key in data_json.keys():
            df_dict[int(key)]["Choosen option"] = data_json[key]["Choosen option"]
    #Convert back to dataframe
    df = pd.DataFrame(list(df_dict.values()))
    return df
# Call the function with df_source
df_processed = choose_best_match_GPT(df_source)
#Merge df_source with df_processed based on index
df_results = df_source.merge(df_processed, left_index=True, right_index=True)
# Save updated DataFrame to Excel
df_results.to_excel(output_path, index=False)

