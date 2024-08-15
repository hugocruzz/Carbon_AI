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
from functions import load_global_env, get_file_paths, pre_process_source_df, update_dataframe_with_correction, hierarchical_selection, emphasize_and_combine_columns, df_to_hyper, assign_columns
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
            df.to_hdf(translate_file, key='df', mode='w', complevel=9, complib='blosc')
        elif columns_to_translate:
            df = pd.read_hdf(translate_file, key='df')
        else:
            df = embed_dataframe(df, columns_to_embed, combined_column_name="combined",
                                output_embedding_name="embedding", api_key=api_key, embedding_model="text-embedding-3-small")
            df.to_hdf(embed_file, key='df', mode='w', complevel=9, complib='blosc')
    else:
        df = pd.read_hdf(embed_file, key='df')
    
    return df


def main(reset: bool = False, semantic_error_estimation: bool = True):
    
    logging.basicConfig(level=logging.INFO)

    # Load the API key
    load_global_env()
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI')
    os.environ["FRED_API_KEY"] = os.getenv('FRED')

    # Load configuration
    config = load_config("scripts/config.yaml")
    paths = config.data_paths
    columns = config.columns
    currency_settings = config.currency_settings
    automated_column_labeling = columns["automated_column_labeling"]
    #IS it possible to set a bool in a yaml file?
    automated_column_labeling: bool = config.automated_column_labeling

    paths["source_translated_file"] = get_file_paths(paths["source_file"], "_translated.h5")
    paths["source_embedded_file"] = get_file_paths(paths["source_file"], "_embedded.h5")
    paths["target_translated_file"] = get_file_paths(paths["target_file"], "_translated.h5")
    paths["target_embedded_file"] = get_file_paths(paths["target_file"], "_embedded.h5")

    try:
        if reset or not os.path.exists(paths["source_embedded_file"]):
            # Process source DataFrame
            logging.info("Processing source DataFrame")

            source_df = pd.read_excel(paths["source_file"])
            SV_unit = pd.read_excel(r"data\achats_EPFL\SV_units.xlsx")[["Unité Organisationnelle", "Institut"]]
            source_df = source_df.merge(SV_unit, on="Unité Organisationnelle", how="left")

            if automated_column_labeling:       
                columns = assign_columns(os.environ["OPENAI_API_KEY"], columns, source_df.drop(columns=columns["source_confidential_column"], errors='ignore')) #Drop column because confidential information 

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
                                            paths["source_embedded_file"], os.environ["OPENAI_API_KEY"])
        else:
            source_df = pd.read_hdf(paths["source_embedded_file"], key='df')
            SV_unit = pd.read_excel(r"data\achats_EPFL\SV_units.xlsx")[["Unité Organisationnelle", "Institut"]]
            source_df = source_df.merge(SV_unit, on="Unité Organisationnelle", how="left")
            if automated_column_labeling:
                columns = assign_columns(os.environ["OPENAI_API_KEY"], columns, source_df.drop(columns=columns["source_confidential_column"], errors='ignore'))

            source_df = emphasize_and_combine_columns(source_df, columns["source_columns_emphasis"], columns["source_columns_to_embed"])

        if reset or not os.path.exists(paths["target_embedded_file"]):
            # Process target DataFrame
            logging.info("Processing target DataFrame")
            target_df = pd.read_excel(paths["target_file"])
            target_df[columns["target_columns_to_embed"]]= target_df[columns["target_columns_to_embed"]].apply(lambda x: x.str.rstrip())

            target_df = translate_and_embed(target_df, columns["target_columns_to_translate"], 
                                            columns["target_columns_to_embed"], paths["target_translated_file"], 
                                            paths["target_embedded_file"], os.environ["OPENAI_API_KEY"])
        else:
            target_df = pd.read_hdf(paths["target_embedded_file"], key='df')
            target_df = emphasize_and_combine_columns(target_df, columns["target_columns_emphasis"], columns["target_columns_to_embed"])
            target_df = target_df.drop_duplicates(subset='combined', keep='first').reset_index(drop=True)


        # Match the datasets
        logging.info("Matching datasets")
        matched_df = match_datasets(source_df, target_df, gpt_model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"], top_n=10)

        converted_currency_serie, correct_inflation_serie = currency_converter(matched_df, 
                                          target_currency=currency_settings["target_currency"], 
                                          date_column=columns["date_column"], 
                                          currency_column=columns["currency_column"], 
                                          amount_column=columns["amount_column"], 
                                          target_inflation_year=currency_settings["target_inflation_year"], 
                                          fred_api_key=os.environ["FRED_API_KEY"])
        converted_CHF_currency_serie, correct_CHF_inflation_serie = currency_converter(matched_df, 
                                          target_currency="CHF", 
                                          date_column=columns["date_column"], 
                                          currency_column=columns["currency_column"], 
                                          amount_column=columns["amount_column"], 
                                          target_inflation_year=currency_settings["target_inflation_year"], 
                                          fred_api_key=os.environ["FRED_API_KEY"])

        df_converted = matched_df.copy()
        df_converted[f'Amount in {currency_settings["target_currency"]}'] = converted_currency_serie
        df_converted[f'Amount in {currency_settings["target_currency"]} corrected inflation in {currency_settings["target_inflation_year"]}'] = correct_inflation_serie
        df_converted[f'Amount in CHF'] = converted_CHF_currency_serie


        df_converted["CO2e"] = df_converted[f'Amount in {currency_settings["target_currency"]} corrected inflation in {currency_settings["target_inflation_year"]}'] * df_converted[columns["emission_factor_column"]]
        # Calculate the total uncertainty factor for each row
        df_converted['total_uncertainty_factor'] = df_converted['per1p5.uncertainty.attr.kg.co2e.per.euro'] + \
                                        (0.8 * df_converted['per1p5.uncertainty.80pct.kg.co2e.per.euro'])

        # Calculate the uncertainty in carbon emissions for each row
        df_converted['uncertainty_carbon_emissions'] = df_converted['CO2e'] * df_converted['total_uncertainty_factor']

        columns_to_keep = list(source_df.columns) + [
            "combined_target", 'total_uncertainty_factor', 'uncertainty_carbon_emissions', 'Amount in CHF', f'Amount in {currency_settings["target_currency"]}', 
            f'Amount in {currency_settings["target_currency"]} corrected inflation in {currency_settings["target_inflation_year"]}', "CO2e"
        ] + columns["target_columns_to_keep"]

        final_df = df_converted[columns_to_keep].drop(columns=["embedding"])
        #Filter final_df with "Entité de gestion"=="002000School of Life Sciences", extract only ["Unité Organisationnelle", "Entité de gestion"], make them unique depending on 
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
    main(reset=True, semantic_error_estimation=True)
