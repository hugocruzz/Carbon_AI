import os
import pandas as pd
from openai import OpenAI
import time
import logging
import json
from openai import OpenAIError
from functions import *

# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize the OpenAI client with your API key

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
                    {"role": "system", "content": f"""
                        You are an assistant that evaluates the semantic similarity between two texts.
                        For each element in the provided JSON dictionary, rate the semantic similarity between the 'Article name' and 'Option' on a scale from 1 to 5, where 1 means completely unrelated and 5 means highly related.
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

def check_semantic_similarity_batch(df, combined_col='combined', target_col='combined_target', batch_size=10, model='gpt-4'):
    text_pairs = df[[combined_col, target_col]].values.tolist()
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    flags = []
    rating = []
    for i in range(0, len(text_pairs), batch_size):
        batch_pairs = text_pairs[i:i+batch_size]
        chunk = {j: {"Article name": pair[0], "Option": pair[1]} for j, pair in enumerate(batch_pairs)}
        
        if len(chunk) == 1:
            example_json = {0: {"Semantic match": 1}}
        else:
            example_json = {0: {"Semantic match": 1}, 1: {"Semantic match": 4}}

        try:
            data_json = call_api_with_retries(client, model, chunk, example_json)
        except Exception as e:
            logging.error(f"Failed to retrieve data: {e}")
            flags.extend([False] * len(chunk))
            continue
        rating.extend([item["Semantic match"] for item in data_json.values()])
        flags.append([item["Semantic match"] == 1 for item in data_json.values()])
    # Flatten the list of lists
    flags = [item for sublist in flags for item in sublist]
    df['Flag'] = pd.Series(flags, index=df.index)
    df['Rating'] = pd.Series(rating, index=df.index)
    return df

def find_semantic_mismatches_batch(df, combined_col='combined', target_col='combined_target', batch_size=10, model='gpt-4o-mini'):
    df_unique = df.drop_duplicates(subset=["combined"])
    df_result = check_semantic_similarity_batch(df_unique, combined_col, target_col, batch_size, model)
    # Select only the new columns to merge, keeping "combined" as the key
    new_columns = [col for col in df_result.columns if col not in df.columns or col == "combined"]

    # Merge only with the selected new columns, avoiding duplicates
    df_merged = df.merge(df_result[new_columns], on="combined", how="left")
    return df_merged

# Example usage
if __name__ == "__main__":
    # Load your data
    source_path =r"data\Results\EPFL_CO2_2023.xlsx"
    df = pd.read_excel(source_path)
    
    # Process the DataFrame in batches and add the 'Flag' column
    df_merged = find_semantic_mismatches_batch(df, batch_size=200)

    # Optionally, save the result to a file
    output_path = source_path.replace(".xlsx", "_checked.xlsx")
    df_merged.to_excel(output_path, index=False)

