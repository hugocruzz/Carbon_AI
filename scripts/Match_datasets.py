import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import json 
from difflib import get_close_matches
import time
import json
import logging
from openai import OpenAIError

def call_api_with_retries(client, model, chunk, example_json, max_retries=5, initial_delay=1):
    success = False
    retries = 0
    delay = initial_delay

    while not success and retries < max_retries:
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content":
                     f"""You are an assistant tasked with selecting the most relevant option from a list based on an 'Article name' and its description. 
                        For each element in the provided JSON dictionary, prioritize matching the text inside asterisks (*) within the 'Article name' to the 'Options' field. 
                        Use the rest of the text as contextual information to ensure compatibility. 
                        Provide your output in valid JSON format. 
                        The data schema should be like this: {json.dumps(example_json)}"""
                    },
                    {"role": "user", "content": json.dumps(chunk)}
                ]
            )
            data = chat_completion.choices[0].message.content
            data_json = json.loads(data)
            success = True
        except OpenAIError as e:
            logging.error(f"API call failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
            retries += 1
            delay *= 2  # Exponential backoff
        except json.JSONDecodeError as e:
            logging.error(f"JSON decode error: {e}")
            break
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            break

    if success:
        return data_json
    else:
        raise Exception("Failed to complete API call after multiple retries")


def Match_datasets(df_source, df_target, top_n=10, output_path=None, gpt_model="gpt-3.5-turbo", api_key=None):
    """Match source and target datasets using embeddings and GPT model."""

    def mapping_embeddings(df_source, df_target, top_n=10):
        source_embedding = np.array(df_source['embedding'].tolist())
        target_embedding = np.array(df_target['embedding'].tolist())
        similarity_matrix = cosine_similarity(source_embedding, target_embedding)
        closest_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_n]
        max_similarity_scores = np.sort(similarity_matrix, axis=1)[:, -top_n:][:, ::-1]
        return max_similarity_scores, closest_indices
    
    df_unique = df_source.drop_duplicates(subset=["combined"])

    if top_n == 1:
        similarity_scores, closest_indices = mapping_embeddings(df_unique, df_target, top_n)
        df_unique["Similarity_scores"] = list(similarity_scores)
        df_unique["combined_target"] = [[df_target.loc[idx, "combined"] for idx in row][0] for row in closest_indices]
        Warning("Top_n is set to 1, only the closest match is returned, no need for GPT to choose the best match.")
        return df_unique
    else:
        similarity_scores, closest_indices = mapping_embeddings(df_unique, df_target, top_n)
        df_unique["Similarity_scores"] = list(similarity_scores)
        df_unique["combined_target"] = [[df_target.loc[idx, "combined"] for idx in row] for row in closest_indices]
    
    os.environ["OPENAI_API_KEY"] = api_key
    df = df_unique.copy()
    df.rename(columns={"combined":'Article name', "combined_target": "Options"}, inplace=True)
    df_dict = df[["Article name", "Options"]].to_dict(orient='records')

    def choose_best_match_GPT(df_dict, model="gpt-3.5-turbo-0125", chunk_size=7):
        client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        df_dict = {i: row for i, row in enumerate(df_dict)}
        df_dict_chunks = [dict(list(df_dict.items())[i:i + chunk_size]) for i in range(0, len(df_dict), chunk_size)]

        for chunk in df_dict_chunks:
            if len(chunk) == 1:
                example_json = {0:{"Chosen option": chunk[list(chunk.keys())[0]]["Options"][0]}}
            else:
                example_json = {0:{"Chosen option": chunk[list(chunk.keys())[0]]["Options"][0]}, 1:{"Chosen option": chunk[list(chunk.keys())[1]]["Options"][0]}}
            #Use a Try and except, if goes to except, repeat the try process
            try:
                data_json = call_api_with_retries(client, model, chunk, example_json)
            except Exception as e:
                print(f"Failed to retrieve data: {e}")
                continue
            for key in data_json.keys():
                df_dict[int(key)]["Chosen option"] = data_json[key]["Chosen option"]
        df = pd.DataFrame(list(df_dict.values()))
        return df

    df_processed = choose_best_match_GPT(df_dict, model=gpt_model)

    df_results = df_unique.reset_index(drop=True).merge(df_processed, left_index=True, right_index=True)
    df_results.rename(columns={"Chosen option": "combined_source_gpt"}, inplace=True)
    df_target.rename(columns={"combined": "combined_target"}, inplace=True)
    df_matched = pd.merge(df_results, df_target, left_on="combined_source_gpt", right_on="combined_target", how="left")
    unmatched_mask = df_matched["combined_target_y"].isnull()
    unmatched_df = df_matched[unmatched_mask].copy()

    if not unmatched_df.empty:
        def find_closest_match(source_value, target_values):
            matches = get_close_matches(source_value, target_values, n=1, cutoff=0.6)
            return matches[0] if matches else None
        unmatched_not_none = unmatched_df["combined_source_gpt"].loc[~unmatched_df["combined_source_gpt"].isnull()]

        unmatched_df["combined_source_gpt"] = unmatched_not_none.apply(lambda x: find_closest_match(x, df_target["combined_target"].tolist()))
        re_matched = pd.merge(unmatched_df, df_target, left_on="combined_source_gpt", right_on="combined_target", how="left")
        df_matched[unmatched_mask] = re_matched

    df_matched.drop(columns=["combined_source_gpt", "combined_target_x", "Article name", "embedding_x", "embedding_y"], inplace=True)
    df_matched.rename(columns={"combined_target_y": "combined_target"}, inplace=True)
    if output_path:
        df_matched.to_excel(output_path, index=False)

    df_final = pd.merge(df_source, df_matched, on="combined", suffixes=('', '_unique'))
    return df_final

if __name__ == "__main__":
    with open(r"C:\Users\cruz" + r'\API_openAI.txt', 'r') as f:
        read_api_key = f.readline()

    source_path = r"data\Results\test_dataset.pkl"
    target_path = r"data\NACRES_embedded.pkl"
    output_path = r"data\Results/test_translated_gpt_o.xlsx"
    Match_datasets(api_key=read_api_key)
