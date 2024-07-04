import os
import pandas as pd
from Translate_DB import translate_DB
from Embed_dataframe import embed_dataframe
from Match_datasets import Match_datasets
import logging
import json
from typing import List

def load_api_key(api_key_path: str) -> str:
    with open(api_key_path, 'r') as f:
        return f.readline().strip()

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_file_paths(base_path: str, suffix: str) -> str:
    return base_path.replace("." + base_path.split(".")[-1], suffix)

def translate_and_embed(df: pd.DataFrame, columns_to_translate: List[str], columns_to_embed: List[str], translate_file: str, embed_file: str, api_key: str) -> pd.DataFrame:
    if not os.path.exists(embed_file):
        if not os.path.exists(translate_file):
            df = translate_DB(df, columns_to_translate, output_file_name=translate_file)
        else:
            df = pd.read_pickle(translate_file)
        df = embed_dataframe(df, columns_to_embed, combined_column_name="combined", output_embedding_name="embedding", api_key=api_key, output_file_name=embed_file)
    else:
        df = pd.read_pickle(embed_file)
    return df

def main():
    setup_logging()
    
    # Load the API key
    api_key_path = r"C:\Users\cruz\API_openAI.txt"
    api_key = load_api_key(api_key_path)
    
    # Set paths
    source_path = r"data\achats_EPFL/Test_100_articles.xlsx"
    target_path = r'data\Raw\PER1p5_nacres_fe_database_v1-0-2023.xlsx'

    source_translated_file = get_file_paths(source_path, "_translated.pkl")
    source_embedded_file = get_file_paths(source_path, "_embedded.pkl")
    target_translated_file = get_file_paths(target_path, "_translated.pkl")
    target_embedded_file = get_file_paths(target_path, "_embedded.pkl")
    output_path = r"data\Results\matched_datasets.xlsx"

    try:
        # Process source DataFrame
        logging.info("Processing source DataFrame")
        source_df = pd.read_excel(source_path)
        #Preprocessing
        source_df = source_df.dropna(subset=["Désignation article"])

        source_columns_to_translate = ["Désignation article", "Famille"]  # Define as needed
        source_columns_to_embed = source_columns_to_translate
        source_df = translate_and_embed(source_df, source_columns_to_translate, source_columns_to_embed, source_translated_file, source_embedded_file, api_key)

        # Process target DataFrame
        logging.info("Processing target DataFrame")
        target_df = pd.read_excel(target_path, sheet_name="NACRES-EF")
        target_df["nacres.description.en"] = target_df["nacres.description.en"].str.rstrip()
        target_columns_to_translate = ["nacres.description.en", "ademe.description", "useeio.description", "module", "category"]  # Define as needed
        target_columns_to_embed = target_columns_to_translate
        target_df = translate_and_embed(target_df, target_columns_to_translate, target_columns_to_embed, target_translated_file, target_embedded_file, api_key)

        # Match the datasets
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
