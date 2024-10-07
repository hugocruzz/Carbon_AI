import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import json
import time
import logging
from openai import OpenAIError

import tiktoken
import math


def call_api_with_retries(client, model, chunk, example_json, max_retries=5, initial_delay=1):
    """
    Calls the OpenAI API with retries in case of failure.

    :param client: OpenAI client
    :param model: Model name to use for the API call
    :param chunk: Data chunk to process
    :param example_json: Example JSON for formatting
    :param max_retries: Maximum number of retries
    :param initial_delay: Initial delay between retries
    :return: Parsed JSON data from the API response
    """
    success = False
    retries = 0
    delay = initial_delay
    example_json = {0:{"ID": 33, "Chosen option": "ELECTRONIC : COMPONENTS, EQUIPMENT"}, 29:{"ID": 1472,"Chosen option": "CONSUMABLES, FURNITURE AND OFFICE EQUIPMENT"}}
    while not success and retries < max_retries:
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                #Integrer dans ll'exemple des choses qui ne fonctionnent pas, des exemples positifs et negatifs 
                messages=[
                    {"role": "system", "content":
                     f"""You are an assistant tasked with selecting the most relevant option in the "Options" field from a list based on an 'Article name' and its description.
                        Additionnaly, you have the ID of the "Options" provided, the output should be the corresponding ID of the chosen option and the content of the Chosen option.
                        If the 'Article name' mentions multiple categories (e.g., biological, chemical, and gaseous or Laboratory, measurement, observation and testing equipment), 
                        prioritize options that broadly cover all or most of these categories, rather than focusing on specific terms.
                        Choose the option that best represents a broad category over a specific one, unless the context strongly favors specificity.

                        GOOD example:
                        Suppose that the article name is "LENOVO ThinkPad Ps G - i GB RAM (xGB =  free socket) TB SSD WUXGA (x) ", this article has an index 53,
                          the output matched ID would be 471, and the chosen option should be  "*PORTABLE COMPUTERS* Computer terminals and other computer peripheral equipment devices".
                         471 being the ID corresponding to the chosen option description, 53 is the index of the data in the input list.

                        Provide your output in valid JSON format.
                        The data schema should be like this: {json.dumps(example_json)}.
                        The output length should be the same as the input length, the index should be the same as the input index.

                        The first key is the index of the data in the input list, for each of them there is a dictionnary containing the ID of the chosen option and the description of the chosen option.
                    """
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

def estimate_chunk_size(df_dict, token_limit=120000, model="gpt-4o-mini"):
    """
    Estimate the chunk size for df_dict such that the number of tokens per chunk is below token_limit.
    
    Parameters:
    df_dict (dict): Dictionary with 2 keys. Each key's value should be a string or similar tokenizable data.
    token_limit (int): The maximum number of tokens allowed per chunk. Default is 120000.
    model (str): The model to use for tokenization (default is gpt-3.5-turbo).
    
    Returns:
    int: The calculated chunk size.
    """

    # Initialize the tokenizer for the specified model
    tokenizer = tiktoken.encoding_for_model(model)

    # Calculate total tokens for each key's value in the dictionary
    total_tokens = sum(
        len(tokenizer.encode(inner_value))
        for inner_dict in df_dict.values() if isinstance(inner_dict, dict)
        for inner_value in inner_dict.values() if isinstance(inner_value, str)
    )
    
    # Estimate chunk size
    chunk_size = max(1, math.floor(token_limit / total_tokens))
    
    return chunk_size
def calculate_similarity_embeddings(df_source, df_target, top_n=10):
    """
    Calculates the similarity between source and target datasets using embeddings.

    :param df_source: Source DataFrame
    :param df_target: Target DataFrame
    :param top_n: Number of top similarities to consider
    :return: Similarity scores and indices of closest matches
    """
    source_embeddings = np.array(df_source['embedding'].tolist())
    target_embeddings = np.array(df_target['embedding'].tolist())
    similarity_matrix = cosine_similarity(source_embeddings, target_embeddings)
    closest_indices = np.argsort(-similarity_matrix, axis=1)[:, :top_n]
    max_similarity_scores = np.sort(similarity_matrix, axis=1)[:, -top_n:][:, ::-1]
    return max_similarity_scores, closest_indices

def prepare_data(df_source, df_target, top_n):
    """
    Prepares the source and target datasets for matching.

    :param df_source: Source DataFrame
    :param df_target: Target DataFrame
    :param top_n: Number of top similarities to consider
    :return: Prepared source and target DataFrames
    """
    df_unique = df_source.drop_duplicates(subset=["combined"]).copy()
    similarity_scores, closest_indices = calculate_similarity_embeddings(df_unique, df_target, top_n)
    df_unique["similarity_scores"] = list(similarity_scores)
    df_unique["combined_target"] = [[df_target.loc[idx, "combined"] for idx in row] for row in closest_indices]
    df_unique["closest_indices"] = list(closest_indices)

    if "embedding" in df_unique.columns:
        df_unique.drop(columns=["embedding"], inplace=True)
    if "embedding" in df_target.columns:
        df_target.drop(columns=["embedding"], inplace=True)
    df_target.rename(columns={"combined": "combined_target"}, inplace=True)
    return df_unique, df_target

def choose_best_match_gpt(df_dict, model="gpt-3.5-turbo-0125", chunk_size=20):
    """
    Chooses the best match using GPT based on similarity scores.

    :param df_dict: Dictionary of data to be processed
    :param model: Model name to use for the GPT
    :param chunk_size: Size of the chunks to process at a time
    :return: DataFrame with chosen options
    """
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    chunk_size = estimate_chunk_size(df_dict, model=model)
    df_dict_chunks = [
        dict(list(df_dict.items())[i:i + chunk_size]) for i in range(0, len(df_dict), chunk_size)
    ]
    output_json = {}
    for chunk in df_dict_chunks:
        if len(chunk) == 1:
            example_json = {
                0: {"Chosen option": chunk[list(chunk.keys())[0]]["Options"][0]}
            }
        else:
            example_json = {"4", "147"}

        try:
            data_json = call_api_with_retries(client, model, chunk, example_json)
        except Exception as e:
            logging.error(f"Failed to retrieve data: {e}")
            continue
        #Get the chosen option with the data_json results
        output_json.update(data_json)

    return output_json


def match_datasets(df_source, df_target, top_n=10, gpt_model="gpt-3.5-turbo", api_key=None, chunk_size=20):
    """
    Matches source and target datasets using embeddings and GPT model.

    :param df_source: Source DataFrame
    :param df_target: Target DataFrame
    :param top_n: Number of top similarities to consider
    :param gpt_model: GPT model to use for matching
    :param api_key: API key for OpenAI
    :return: DataFrame with matched results
    """
    os.environ["OPENAI_API_KEY"] = api_key

    # Step 1: Prepare data by dropping duplicates and handling embeddings
    df_unique, df_target = prepare_data(df_source, df_target, top_n)
    df_unique.reset_index(drop=True, inplace=True)
    # Step 2: Rename columns for consistency and prepare the dictionary for GPT
    df_unique.rename(columns={"combined": 'Article name', "combined_target": "Options", "closest_indices": "ID"}, inplace=True)
    df_unique["ID"] = df_unique["ID"].apply(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

    df_dict = df_unique[["Article name", "Options", "ID"]].to_dict(orient='index')
    df_unique.drop(columns=["similarity_scores", "Options"], inplace=True)

    # Step 3: Use GPT to choose the best match
    df_dict_processed = choose_best_match_gpt(df_dict, model=gpt_model, chunk_size=chunk_size)
    #Extract the "ID" value for each key in the df_dict_matched_index dictionnary
    df_dict_matched_index = {int(k): v["ID"] for k, v in df_dict_processed.items()}
    df_dict_matched_series = pd.Series(df_dict_matched_index)
    mapped_series = df_dict_matched_series.map(df_target["combined_target"])
    df_dict_matched_series.index = df_dict_matched_series.index.astype(int)
    mapped_series.index = mapped_series.index.astype(df_unique.index.dtype)
    df_unique["combined_target"] = mapped_series
    # Step 4: Merge results and rename columns
    df_unique.rename(columns={"Article name": "combined"}, inplace=True)
    #Merge df_source with df_unique based on the "combined" column
    df_original = pd.merge(df_source, df_unique, on="combined", suffixes=('', '_unique'))
    #Drop all columns that ends on "_unique"
    df_original.drop(columns=[col for col in df_original.columns if col.endswith("_unique")], inplace=True)

    #Merge df_orginal with df_target based on the "combined_target" column
    df_final = pd.merge(df_original, df_target, left_on="combined_target", right_on="combined_target", how="left")
    return df_final