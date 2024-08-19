import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
import json
import time
import logging
from difflib import get_close_matches
from openai import OpenAIError


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

    while not success and retries < max_retries:
        try:
            chat_completion = client.chat.completions.create(
                model=model,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content":
                     f"""You are an assistant tasked with selecting the most relevant option in the "Options" field from a list based on an 'Article name' and its description.
                        For each element in the provided JSON dictionary, your primary goal is to select the option that represents the broadest category encompassing the key terms found inside asterisks (*) within the 'Article name'.
                        If the 'Article name' mentions multiple categories (e.g., biological, chemical, and gaseous), prioritize options that broadly cover all or most of these categories, rather than focusing on specific terms.
                        Choose the option that best represents a broad category over a specific one, unless the context strongly favors specificity.
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
    df_unique = df_source.drop_duplicates(subset=["combined"])
    similarity_scores, closest_indices = calculate_similarity_embeddings(df_unique, df_target, top_n)

    df_unique["similarity_scores"] = list(similarity_scores)
    df_unique["combined_target"] = [[df_target.loc[idx, "combined"] for idx in row] for row in closest_indices]

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
    df_dict_chunks = [
        dict(list(df_dict.items())[i:i + chunk_size]) for i in range(0, len(df_dict), chunk_size)
    ]

    for chunk in df_dict_chunks:
        if len(chunk) == 1:
            example_json = {
                0: {"Chosen option": chunk[list(chunk.keys())[0]]["Options"][0]}
            }
        else:
            example_json = {
                0: {"Chosen option": chunk[list(chunk.keys())[0]]["Options"][0]},
                1: {"Chosen option": chunk[list(chunk.keys())[1]]["Options"][0]}
            }

        try:
            data_json = call_api_with_retries(client, model, chunk, example_json)
        except Exception as e:
            logging.error(f"Failed to retrieve data: {e}")
            continue

        for key in data_json.keys():
            df_dict[int(key)]["Chosen option"] = data_json[key]["Chosen option"]

    return pd.DataFrame(list(df_dict.values()))

def handle_unmatched_cases(df_matched, df_target, gpt_model="gpt-3.5-turbo-0125", chunk_size=20):
    """
    Handles cases where matches are not found initially by using `find_closest_match` first
    and then using GPT model for remaining unmatched cases.

    :param df_matched: DataFrame of matched results
    :param df_target: Target DataFrame
    :param gpt_model: GPT model to use for matching
    :return: DataFrame with unmatched cases handled
    """

    attempt = 0
    
    unmatched_mask = df_matched["combined_target"].isnull()
    unmatched_df = df_matched[unmatched_mask]
    while not unmatched_df.empty and attempt < 3:
        # Identify unmatched cases
        unmatched_df.drop(columns=df_target.columns, inplace=True)

        # Step 1: Use find_closest_match for entries that might have been partially matched
        unmatched_df["combined_target_gpt"] = unmatched_df["combined_target_gpt"].apply(
            lambda x: find_closest_match(x, df_target["combined_target"].tolist()) if pd.notnull(x) else x)
        
        #Merge the data 
        df_unmatched_merged = pd.merge(unmatched_df, df_target, left_on="combined_target_gpt", right_on="combined_target", how="left")
        df_unmatched_merged.set_index(unmatched_df.index, inplace=True)
        df_matched.update(df_unmatched_merged)

        ########## GPT processing ##########
        unmatched_mask = df_matched["combined_target"].isnull()
        unmatched_df = df_matched[unmatched_mask].copy()

        if unmatched_df.empty:
            break

        unmatched_df.drop(columns=["combined_target_gpt", "Article name"], inplace=True)
        unmatched_df.drop(columns=df_target.columns, inplace=True)
        # Prepare data for GPT processing
        unmatched_df.rename(columns={"combined": 'Article name'}, inplace=True)
        df_dict_unmatched = unmatched_df[["Article name", "Options"]].to_dict(orient='index')

        # Call GPT to attempt to find matches
        df_processed_unmatched = choose_best_match_gpt(df_dict_unmatched, model=gpt_model, chunk_size=chunk_size)
        df_processed_unmatched.set_index(unmatched_df.index, inplace=True)
        unmatched_df["combined_target_gpt"] = df_processed_unmatched["Chosen option"]

        # Merge the newly matched entries back into the main DataFrame
        df_unmatched_merged = pd.merge(unmatched_df, df_target, left_on="combined_target_gpt", right_on="combined_target", how="left")
        df_unmatched_merged.set_index(unmatched_df.index, inplace=True)
        
        df_matched.update(df_unmatched_merged)
        # Re-check which entries still haven't been matched after both methods
        unmatched_mask = df_matched["combined_target"].isnull()
        unmatched_df = df_matched[unmatched_mask]
        
        attempt += 1

    return df_matched


def find_closest_match(source_value, target_values):
    """
    Finds the closest match from the target values.

    :param source_value: Source value to find a match for
    :param target_values: List of target values to search
    :return: Closest match or None
    """
    matches = get_close_matches(source_value, target_values, n=1, cutoff=0.6)
    return matches[0] if matches else None

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
    df_unique.rename(columns={"combined": 'Article name', "combined_target": "Options"}, inplace=True)
    df_dict = df_unique[["Article name", "Options"]].to_dict(orient='index')
    df_unique.drop(columns=["similarity_scores", "Options"], inplace=True)

    # Step 3: Use GPT to choose the best match
    df_processed = choose_best_match_gpt(df_dict, model=gpt_model, chunk_size=chunk_size)

    # Step 4: Merge results and rename columns
    df_unique.rename(columns={"Article name": "combined"}, inplace=True)
    df_results = df_unique.reset_index(drop=True).merge(df_processed, left_index=True, right_index=True)
    df_results.rename(columns={"Chosen option": "combined_target_gpt"}, inplace=True)

    # Step 5: Initial matching with target dataset
    df_matched = pd.merge(df_results, df_target, left_on="combined_target_gpt", right_on="combined_target", how="left")

    # Step 6: Handle unmatched cases
    df_matched = handle_unmatched_cases(df_matched, df_target, gpt_model=gpt_model, chunk_size=chunk_size)

    # Step 7: Final merge with the source DataFrame
    df_final = pd.merge(df_source, df_matched, on="combined", suffixes=('', '_unique'))
    #Drop all columns that ends on "_unique"
    df_final.drop(columns=[col for col in df_final.columns if col.endswith("_unique")], inplace=True)

    return df_final