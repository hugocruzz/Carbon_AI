
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import json 

gpt_model ="gpt-4o"

with open(r"C:\Users\cruz" + r'\API_openAI.txt', 'r') as f:
    read_api_key = f.readline()
    os.environ["OPENAI_API_KEY"] = read_api_key

path_source = r"data\Results\test_dataset.pkl"
output_path = r"data\Results/test_translated_gpt_o.xlsx"
database_path =r"data\NACRES_embedded.pkl"

df_source = pd.read_pickle(path_source)
df_source.rename(columns={'embedding_y': 'embedding'}, inplace=True)
#Drop rows where df_source["Désignation article"] is nan
df_source = df_source.dropna(subset=["Désignation article"])
df_target = pd.read_pickle(database_path)
#IF df_target["nacres.description.en"] ends with " " remove it
df_target["nacres.description.en"] = df_target["nacres.description.en"].str.rstrip()
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
df_source["NACRES_code"] = [[df_target.loc[idx, "nacres.code"] for idx in row] for row in closest_indices]

df_embedded = df_source.copy()
df_embedded["NACRES_names"] = [[df_target.loc[idx, "nacres.description.en"] for idx in row][0] for row in closest_indices]
df_embedded["NACRES_code"] = [[df_target.loc[idx, "nacres.code"] for idx in row][0] for row in closest_indices]
df_embedded.to_excel(r"data\Results/achats_test_embedding.xlsx", index=False)

df = df_source.copy()
df.rename(columns={'Désignation article': 'Article name', 'Famille':'Category', "Fournisseur": "Provider", "NACRES_names": "Options", "Prix unitaire": "Price"}, inplace=True)
#COnvert df["Price"] to str and add "€" at the end
df["Price"] = df["Price"].astype(str) + "€"
df_dict = df[["Article name", "Category", "Options", "Price"]].to_dict(orient='records')

def choose_best_match_GPT(df_dict, model="gpt-3.5-turbo-0125", chunk_size = 7):
    #Chunk df_dict into smaller chunks
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    example_json = {1:{"Choosen option": 'CLINICAL INVESTIGATION CONSUMABLES AND REAGENTS'}, 2:{"Choosen option": 'CELL BIOLOGY: ASSAY KITS, FUNCTIONAL ASSAYS - BIOCHEMICAL KITS'}}
    
    # Convert list of dictionaries into a dictionary with indices as keys
    df_dict = {i: row for i, row in enumerate(df_dict)}

    #df_dict is a dictionnary, split it into chunks of 10
    df_dict_chunks = [dict(list(df_dict.items())[i:i + chunk_size]) for i in range(0, len(df_dict), chunk_size)]

    for chunk in df_dict_chunks:
        chat_completion = client.chat.completions.create(
            model=model,
            response_format={"type":"json_object"},
            messages=[
                {"role": "system", "content": """Provide output in valid JSON. Your are given a dictionnary in json format.
                For each elements, and based on the informations you have in the 'Article name', 'Category' and 'Provider' fields, 
                 you need to choose between 1 of the options in the 'Options' field or None of the options if you think that any options you're given does not fit.
                 The choosen option should be the most relevant to the 'Article name', 'Category' and 'Price' fields.
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
df_processed = choose_best_match_GPT(df_dict, model=gpt_model)
#Merge df_source with df_processed based on index
df_results = df_source.merge(df_processed, left_index=True, right_index=True)
# Save updated DataFrame to Excel
df_catalyse_nacres = pd.merge(df_results, df_target, left_on="Choosen option", right_on="nacres.description.en", how="left")
df_catalyse_nacres["unité"] = 1
df_catalyse_nacres["Prix total"] = df_catalyse_nacres["Prix unitaire"] * df_catalyse_nacres["unité"]
df_catalyse_nacres["CO2 kg"] = df_catalyse_nacres["ademe.ef.kg.co2e.per.euro"]*df_catalyse_nacres["Prix total"]
df_catalyse_nacres.to_excel(output_path, index=False)