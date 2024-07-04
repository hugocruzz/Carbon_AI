
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import json 

from difflib import get_close_matches
def Match_datasets(df_source, df_target, top_n=10, output_path=None, gpt_model="gpt-4o", api_key=None):
    
    def mapping_embeddings(df_source, df_target, top_n=10):
        source_embedding = np.array(df_source['embedding'].tolist())
        target_embedding = np.array(df_target['embedding'].tolist())
        similarity_matrix = cosine_similarity(source_embedding, target_embedding)

        # Get indices of the top 10 closest matches
        closest_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_n]
        # Extract the top 10 similarity scores
        max_similarity_scores = np.sort(similarity_matrix, axis=1)[:, -top_n:][:, ::-1]
        return max_similarity_scores, closest_indices
    
    if top_n==1:
        similarity_scores, closest_indices = mapping_embeddings(df_source, df_target, top_n)
        df_source["Similarity_scores"] = list(similarity_scores)
        df_source["combined_target"] = [[df_target.loc[idx, "combined"] for idx in row][0] for row in closest_indices]
        Warning("Top_n is set to 1, only the closest match is returned, no need for GPT to choose the best match.")
        return df_source
    else:
        similarity_scores, closest_indices = mapping_embeddings(df_source, df_target, top_n)
        df_source["Similarity_scores"] = list(similarity_scores)
        df_source["combined_target"] = [[df_target.loc[idx, "combined"] for idx in row] for row in closest_indices]
    
    os.environ["OPENAI_API_KEY"] = api_key
    df = df_source.copy()
    df.rename(columns={"combined":'Article name', "combined_target": "Options"}, inplace=True)
    #df["Price"] = df["Price"].astype(str) + "€"
    df_dict = df[["Article name", "Options"]].to_dict(orient='records')

    def choose_best_match_GPT(df_dict, model="gpt-3.5-turbo-0125", chunk_size = 7):
        #Chunk df_dict into smaller chunks
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        # Convert list of dictionaries into a dictionary with indices as keys
        df_dict = {i: row for i, row in enumerate(df_dict)}

        #df_dict is a dictionnary, split it into chunks of 10
        df_dict_chunks = [dict(list(df_dict.items())[i:i + chunk_size]) for i in range(0, len(df_dict), chunk_size)]

        for chunk in df_dict_chunks:
            #get first element of list chunk
            first_element = list(chunk.keys())[0]
            example_json = {1:{"Chosen option": chunk[list(chunk.keys())[0]]["Options"][0]}, 2:{"Chosen option": chunk[list(chunk.keys())[1]]["Options"][0]}}
            chat_completion = client.chat.completions.create(
                model=model,
                response_format={"type":"json_object"},
                messages=[
                    {"role": "system", "content": """Provide output in valid JSON. Your are given a dictionnary in json format.
                    For each elements, and based on the informations you have in the 'Article name',
                    you need to choose between one of the options in the 'Options' field or None of the options if you think that any options you're given does not fit.
                    The chosen option should be the most relevant to the 'Article name' field.
                    The data schema should be like this :""" + json.dumps(example_json)},
                    {"role": "user", "content": json.dumps(chunk)}
                ]
            )
            data = chat_completion.choices[0].message.content
            data_json = json.loads(data)
            for key in data_json.keys():
                df_dict[int(key)]["Chosen option"] = data_json[key]["Chosen option"]
        #Convert back to dataframe
        df = pd.DataFrame(list(df_dict.values()))
        return df
    # Call the function with df_source
    df_processed = choose_best_match_GPT(df_dict, model=gpt_model)
    df_results = df_source.merge(df_processed, left_index=True, right_index=True)
    df_results.rename(columns={"Chosen option": "combined_source_gpt"}, inplace=True)
    df_target.rename(columns={"combined": "combined_target"}, inplace=True)
    # First attempt to merge directly
    df_matched = pd.merge(df_results, df_target, left_on="combined_source_gpt", right_on="combined_target", how="left")

    # Identify rows where merge was not successful
    unmatched_mask = df_matched["combined_target_y"].isnull()
    unmatched_df = df_matched[unmatched_mask].copy()

    if not unmatched_df.empty:
        def find_closest_match(source_value, target_values):
            matches = get_close_matches(source_value, target_values, n=1, cutoff=0.6)
            return matches[0] if matches else None
        unmatched_df["combined_source_gpt"] = unmatched_df["combined_source_gpt"].apply(lambda x: find_closest_match(x, df_target["combined_target"].tolist()))

        # Reattempt to merge for the unmatched rows
        re_matched = pd.merge(unmatched_df, df_target, left_on="combined_source_gpt", right_on="combined_target", how="left")

        # Combine the successfully matched rows with the re-matched rows
        df_matched[unmatched_mask] = re_matched
    df_matched_original = df_matched.copy()
    df_matched.drop(columns=["combined_source_gpt", "combined_target_x", "Article name", "embedding_x", "embedding_y"], inplace=True)
    df_matched.rename(columns={"combined_target_y": "combined_target"}, inplace=True)
    if output_path:
        df_matched.to_excel(output_path, index=False)
    return df_matched


if __name__=="__main__":
    with open(r"C:\Users\cruz" + r'\API_openAI.txt', 'r') as f:
        read_api_key = f.readline()

    source_path = r"data\Results\test_dataset.pkl"
    target_path =r"data\NACRES_embedded.pkl"

    output_path = r"data\Results/test_translated_gpt_o.xlsx"
    
    Match_datasets(api_key=read_api_key)
