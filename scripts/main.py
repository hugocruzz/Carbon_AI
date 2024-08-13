import os
import pandas as pd
import logging
import yaml
from typing import List
from dataclasses import dataclass

from translate_db import translate_db
from embed_dataframe import embed_dataframe
from match_datasets import match_datasets
from currency_converter import currency_converter
from functions import load_global_env, get_file_paths, pre_process_source_df, update_dataframe_with_correction, hierarchical_selection, emphasize_and_combine_columns, df_to_hyper
from check_matching_errors import find_semantic_mismatches_batch


@dataclass
class Config:
    data_paths: dict
    columns: dict
    currency_settings: dict


def load_config(config_file: str = "config.yaml") -> Config:
    with open(config_file, "r") as file:
        config_data = yaml.safe_load(file)
    return Config(**config_data)


def translate_and_embed(df: pd.DataFrame, 
                        columns_to_translate: List[str], 
                        columns_to_embed: List[str], 
                        translate_file: str, 
                        embed_file: str, 
                        api_key: str) -> pd.DataFrame:
    """Translate and embed specified dataframe columns."""
    if not os.path.exists(embed_file):
        if not os.path.exists(translate_file) and columns_to_translate:
            df = translate_db(df, columns_to_translate)
            df.to_pickle(translate_file)
        else:
            df = pd.read_pickle(translate_file)
        
        df = embed_dataframe(df, columns_to_embed, combined_column_name="combined",
                             output_embedding_name="embedding", api_key=api_key, embedding_model="text-embedding-3-small")
        df.to_pickle(embed_file)
    else:
        df = pd.read_pickle(embed_file)
    
    return df


def main(reset: bool = False, semantic_error_estimation: bool = True):
    # Load configuration
    config = load_config("scripts/config.yaml")

    # Load the API key
    load_global_env()
    api_key = os.getenv('OPENAI')

    # Set paths
    paths = config.data_paths

    # Extract column settings
    columns = config.columns

    # Extract currency settings
    currency_settings = config.currency_settings

    paths["source_translated_file"] = get_file_paths(paths["source_file"], "_translated.pkl")
    paths["source_embedded_file"] = get_file_paths(paths["source_file"], "_embedded.pkl")
    paths["target_tranlated_file"] = get_file_paths(paths["target_file"], "_translated.pkl")
    paths["target_embedded_file"] = get_file_paths(paths["target_file"], "_embedded.pkl")

    try:
        if reset or not os.path.exists(paths["source_embedded_file"]):
            # Process source DataFrame
            logging.info("Processing source DataFrame")

            source_df = pd.read_excel(paths["source_file"])
            from openai import OpenAI
            df_head = source_df.head()
            prompt = f"""You are provided with the following DataFrame head:
                        {df_head}
                        Identify the column that represents the "purchase date," the column that represents the "amount of money," and the columns that describe the purchase. Respond with your findings.
                        Name them respectively: date_column, amount_column, and source_columns_to_embed.
                        """
            client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
            example_json = {0: {"date_column": "Date", "amount_column": "Amount", "source_columns_to_embed": ["Description"]}}
            

            source_df = pre_process_source_df(columns["source_columns_to_embed"], source_df)

            if columns["key_manual_column"]:
                corrected_df = pd.read_excel(paths["source_file_manual_correction"])
                corrected_df = pre_process_source_df(columns["source_columns_to_embed"], corrected_df)
                source_df = update_dataframe_with_correction(source_df, corrected_df, key_column=columns["key_manual_column"])
            else:
                print("No Manual correction implemented. \nPlease provide a key merging column if you want to update the source DataFrame with corrected values.")
            if columns["hierarchical_selection_column_name"]:
                source_df = hierarchical_selection(source_df, columns["source_columns_to_embed"], columns["hierarchical_selection_column_name"])
                source_columns_to_embed = [columns["hierarchical_selection_column_name"]]
            else:
                print("No hierarchical selection implemented. \nPlease provide a column name if you want to perform hierarchical selection.")

            source_df = translate_and_embed(source_df, columns["source_columns_to_translate"], 
                                            source_columns_to_embed, paths["source_translated_file"], 
                                            paths["source_embedded_file"], api_key)
        else:
            source_df = pd.read_pickle(paths["source_embedded_file"])
            source_df = emphasize_and_combine_columns(source_df, columns["source_columns_emphasis"], columns["source_columns_to_embed"])

        if reset or not os.path.exists(paths["target_embedded_file"]):
            # Process target DataFrame
            logging.info("Processing target DataFrame")
            target_df = pd.read_excel(paths["target_file"])
            target_df[columns["target_columns_to_embed"]] = target_df[columns["target_columns_to_embed"]].str.rstrip()
            target_df = translate_and_embed(target_df, columns["target_columns_to_translate"], 
                                            columns["target_columns_to_embed"], paths["target_translated_file"], 
                                            paths["target_embedded_file"], api_key)
        else:
            target_df = pd.read_pickle(paths["target_embedded_file"])
            target_df = emphasize_and_combine_columns(target_df, columns["target_columns_emphasis"], columns["target_columns_to_embed"])

        # Match the datasets
        logging.info("Matching datasets")
        matched_df = match_datasets(source_df, target_df, gpt_model="gpt-4o-mini", api_key=api_key, top_n=10)

        converted_currency_serie, correct_inflation_serie = currency_converter(matched_df, 
                                          target_currency=currency_settings["target_currency"], 
                                          date_column=columns["date_column"], 
                                          currency_column=columns["currency_column"], 
                                          amount_column=columns["amount_column"], 
                                          target_inflation_year=currency_settings["target_inflation_year"], 
                                          fred_api_key=os.getenv('FRED'))

        df_converted = matched_df.copy()
        df_converted[f'Amount in {currency_settings["target_currency"]}'] = converted_currency_serie
        df_converted[f'Amount in {currency_settings["target_currency"]} corrected inflation in {currency_settings["target_inflation_year"]}'] = correct_inflation_serie

        df_converted["CO2e"] = df_converted[f'Amount in {currency_settings["target_currency"]} corrected inflation in {currency_settings["target_inflation_year"]}'] * df_converted[columns["emission_factor_column"]]

        columns_to_keep = list(source_df.columns) + [
            "combined_target",
            f'Amount in {currency_settings["target_currency"]} corrected inflation in {currency_settings["target_inflation_year"]}', "CO2e"
        ] + columns["target_columns_to_keep"]
        final_df = df_converted[columns_to_keep].drop(columns=["embedding"])
        
        if semantic_error_estimation: 
            logging.info("Estimating semantic errors")
            flagged_df = find_semantic_mismatches_batch(final_df, batch_size=200)
            flagged_df.to_excel(paths["output_file"], index=False)
            output_hyper_path = get_file_paths(paths["output_file"], ".hyper")
            df_to_hyper(flagged_df, output_hyper_path)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise


if __name__ == '__main__':
    main(reset=False, semantic_error_estimation=True)
