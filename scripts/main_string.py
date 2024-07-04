import os
import pandas as pd
from Translate_DB import translate_DB
from Embed_dataframe import embed_dataframe
from Match_datasets import Match_datasets
import logging
import json
from typing import List
import yaml

def load_api_key(api_key_path: str) -> str:
    with open(api_key_path, 'r') as f:
        return f.readline().strip()

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_file_paths(base_path: str, suffix: str) -> str:
    return base_path.replace("." + base_path.split(".")[-1], suffix)

def translate_and_embed(input_data, columns_to_translate: List[str], columns_to_embed: List[str], translate_file: str, embed_file: str, api_key: str) -> pd.DataFrame:
    if not os.path.exists(embed_file):
        if not os.path.exists(translate_file):
            df = translate_DB(input_data, columns_to_translate, output_file_name=translate_file)
        else:
            df = pd.read_pickle(translate_file)
        df = embed_dataframe(df, columns_to_embed, combined_column_name="combined", output_embedding_name="embedding", api_key=api_key, output_file_name=embed_file)
    else:
        df = pd.read_pickle(embed_file)
    return df

def main():
    setup_logging()
    
    api_key_path = r"C:\Users\cruz\API_openAI.txt"
    api_key = load_api_key(api_key_path)
    
    source_input = "Ordinateur de bureau HP EliteDesk 800 G6 - Core i5 10500 - 8 Go RAM - 256 Go SSD - Windows 10 Pro"
    
    #Read the input_path.yaml and define the paths
    with open("scripts/input_path.yaml", "r") as file:
        paths = yaml.safe_load(file)

    target_path = paths["target_file"]
    output_path = paths["output_file"]

    source_translated_file = "data/source_string_translated.pkl"
    source_embedded_file = "data/source_string_embedded.pkl"
    target_translated_file = get_file_paths(target_path, "_translated.pkl")
    target_embedded_file = get_file_paths(target_path, "_embedded.pkl")
    output_path = r"data\Results\matched_datasets.xlsx"

    try:
        logging.info("Processing source input")
        source_columns_to_translate = ["description"]  # Define as needed for the string input, if given, imply a translation, if not, no translation will be performed
        source_columns_to_embed = source_columns_to_translate
        source_df = translate_and_embed(source_input, source_columns_to_translate, source_columns_to_embed, source_translated_file, source_embedded_file, api_key)

        logging.info("Processing target DataFrame")
        target_df = pd.read_excel(target_path, sheet_name="NACRES-EF")
        target_df["nacres.description.en"] = target_df["nacres.description.en"].str.rstrip()
        target_columns_to_translate = ["nacres.description.en", "ademe.description", "useeio.description", "module", "category"]
        target_columns_to_embed = target_columns_to_translate
        
        target_df = translate_and_embed(target_df, target_columns_to_translate, target_columns_to_embed, target_translated_file, target_embedded_file, api_key)

        logging.info("Matching datasets")
        matched_df = Match_datasets(source_df, target_df, output_path=output_path, gpt_model="gpt-4o", api_key=api_key)
        matched_df["unité"] = 1
        matched_df["Prix total"] = matched_df["Prix unitaire"] * matched_df["unité"]
        matched_df["CO2 kg"] = matched_df["ademe.ef.kg.co2e.per.euro"]*matched_df["Prix total"]
        matched_df.to_excel(output_path, index=False)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == '__main__':
    main()
