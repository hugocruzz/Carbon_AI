import os
import pandas as pd
from Translate_DB import translate_DB
from Embed_dataframe import embed_dataframe
from Match_datasets import Match_datasets
import logging
import json
from typing import List
import yaml
from currency_converter import currency_converter
from functions import *

def load_api_key(api_key_path: str) -> str:
    """Load API key from a file."""
    with open(api_key_path, 'r') as f:
        return f.readline().strip()

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_file_paths(base_path: str, suffix: str) -> str:
    """Generate file paths with given suffix."""
    return base_path.replace("." + base_path.split(".")[-1], suffix)

def translate_and_embed(df: pd.DataFrame, columns_to_translate: List[str], columns_to_embed: List[str], translate_file: str, embed_file: str, api_key: str) -> pd.DataFrame:
    """Translate and embed dataframe columns."""
    if not os.path.exists(embed_file):
        if not os.path.exists(translate_file):
            df = translate_DB(df, columns_to_translate, output_file_name=translate_file)
        else:
            df = pd.read_pickle(translate_file)
        df = embed_dataframe(df, columns_to_embed, combined_column_name="combined", output_embedding_name="embedding", api_key=api_key, output_file_name=embed_file, embedding_model="text-embedding-3-small")
    else:
        df = pd.read_pickle(embed_file)
    return df

def to_hyper(df: pd.DataFrame, output_path: str):
    """Convert DataFrame to Tableau
    Hyper format."""


def main():
    setup_logging()
    
    #Setup column name to embed:
    source_columns_to_embed = ["fam_dom_order", "fam_fam_order","fam_sfam_order", "fam_ssfam_order"]
    # Load the API key
    api_key_path = r"C:\Users\cruz\API_openAI.txt"
    api_key = load_api_key(api_key_path)
    
    # Set paths
    #Read the input_path.yaml and define the paths
    with open("scripts/input_path.yaml", "r") as file:
        paths = yaml.safe_load(file)
        
    source_path = paths["source_file"]
    target_path = paths["target_file"]
    output_path = paths["output_file"]

    source_translated_file = get_file_paths(source_path, "_translated.pkl")
    source_embedded_file = get_file_paths(source_path, "_embedded.pkl")
    target_translated_file = get_file_paths(target_path, "_translated.pkl")
    target_embedded_file = get_file_paths(target_path, "_embedded.pkl")

    try:
        # Process source DataFrame
        logging.info("Processing source DataFrame")
        source_df = pd.read_excel(source_path)
        source_df = source_df.dropna(subset=source_columns_to_embed,how = 'all')
        # Remove numbers from the specified columns
        for column in source_columns_to_embed:
            source_df[column] = source_df[column].str.replace(r'\d+', '', regex=True)

        source_columns_to_translate = []
        source_df = translate_and_embed(source_df, source_columns_to_translate, source_columns_to_embed, source_translated_file, source_embedded_file, api_key)

        # Process target DataFrame
        logging.info("Processing target DataFrame")
        target_df = pd.read_excel(target_path, sheet_name="NACRES-EF")
        target_df["nacres.description.en"] = target_df["nacres.description.en"].str.rstrip()
        target_columns_to_translate = []
        target_columns_to_embed = ["nacres.description.en", "useeio.description", "module", "category"]
        
        target_df = translate_and_embed(target_df, target_columns_to_translate, target_columns_to_embed, target_translated_file, target_embedded_file, api_key)

        # Match the datasets
        logging.info("Matching datasets")
        matched_df = Match_datasets(source_df, target_df, output_path=output_path, gpt_model="gpt-3.5-turbo", api_key=api_key, top_n=5)
        df_converted = currency_converter(matched_df, target_currency="EUR", date_column="Date de commande", currency_column="Devise", amount_column="PU commande")
        df_converted["CO2e kg"] = df_converted["Amount in EUR"]*df_converted["Qté commande"]*df_converted["per1p5.ef.kg.co2e.per.euro"]
        columns_to_keep = list(source_df.columns)+ ["combined", "per1p5.ef.kg.co2e.per.euro", "Amount in EUR", "CO2e kg", "combined_target"]
        df_to_hyper(df_converted[columns_to_keep], r'data\Results\EPFL_CO2_2023.hyper')
        #Post processing
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == '__main__':
    main()
