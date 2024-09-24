import os
import pandas as pd
import logging
import yaml
from typing import List
from dataclasses import dataclass
import argparse
from dotenv import load_dotenv

from translate_db import translate_db
from embed_dataframe import embed_dataframe
from match_datasets import match_datasets
from currency_converter import currency_converter
from check_matching_errors import find_semantic_mismatches_batch
from functions import (
    load_global_env,
    get_file_paths,
    pre_process_source_df,
    update_dataframe_with_correction,
    hierarchical_selection,
    emphasize_and_combine_columns,
    df_to_hyper,
    assign_columns,
)

# Configure logging
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_path = os.path.join(log_dir, 'app.log')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()
    ]
)


@dataclass
class Config:
    data_paths: dict
    columns: dict
    currency_settings: dict

def load_config(config_file: str = "uploads/config.yaml") -> Config:
    with open(config_file, "r", encoding="utf-8") as file:  # Specify UTF-8 encoding
        config_data = yaml.safe_load(file)
    return Config(**config_data)



def translate_and_embed(df: pd.DataFrame, 
                        columns_to_translate: List[str], 
                        columns_to_embed: List[str], 
                        translate_file: str, 
                        embed_file: str, 
                        api_key: str,
                        export_files: bool) -> pd.DataFrame:
    """Translate and embed specified dataframe columns."""
    if not os.path.exists(embed_file):
        if not os.path.exists(translate_file) and columns_to_translate:
            df = translate_db(df, columns_to_translate)
            if export_files:
                df.to_hdf(translate_file, key='df', mode='w', complevel=9, complib='blosc')
        elif columns_to_translate:
            df = pd.read_hdf(translate_file, key='df')
        else:
            df = embed_dataframe(df, columns_to_embed, combined_column_name="combined",
                                output_embedding_name="embedding", api_key=api_key, embedding_model="text-embedding-3-small")
            if export_files:
                df.to_hdf(embed_file, key='df', mode='w', complevel=9, complib='blosc')
    else:
        df = pd.read_hdf(embed_file, key='df')
    
    return df


def main(config_path="configs/config.yaml", reset: bool = False, semantic_error_estimation: bool = True, matched_output_export: bool = False, export_files: bool = False):

    # Load the API key, feel free to change it 
    env_path = os.path.expanduser('~/global_env/.env')
    load_dotenv(env_path)
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI')
    os.environ["FRED_API_KEY"] = os.getenv('FRED')

    # Load configuration
    config = load_config(config_path)
    paths = config.data_paths
    columns = config.columns
    currency_settings = config.currency_settings

    automated_column_labeling = columns["automated_column_labeling"]

    paths["source_translated_file"] = get_file_paths(paths["source_file"], "_translated.h5")
    paths["source_embedded_file"] = get_file_paths(paths["source_file"], "_embedded.h5")
    paths["target_translated_file"] = get_file_paths(paths["target_file"], "_translated.h5")
    paths["target_embedded_file"] = get_file_paths(paths["target_file"], "_embedded.h5")

    try:
        if reset or not os.path.exists(paths["source_embedded_file"]):
            # Process source DataFrame
            logging.info("Processing source DataFrame")
            source_df = pd.read_excel(paths["source_file"])
            source_df = source_df.drop_duplicates().reset_index(drop=True)

            if automated_column_labeling:
                columns = assign_columns(os.environ["OPENAI_API_KEY"], columns, source_df.drop(columns=columns["source_confidential_column"], errors='ignore')) #Drop column because confidential information 
            #print duplicate rows 


            source_df = pre_process_source_df(columns["source_columns_to_embed"], source_df)

            if columns["key_manual_column"]:
                corrected_df = pd.read_excel(paths["source_file_manual_correction"])
                corrected_df = pre_process_source_df(columns["source_columns_to_embed"], corrected_df)
                source_df = update_dataframe_with_correction(source_df, corrected_df, key_column=columns["key_manual_column"])
            else:
                print("No Manual correction implemented. Please provide a key merging column if you want to update the source DataFrame with corrected values.")

            if columns["hierarchical_selection_column_name"]:
                source_df = hierarchical_selection(source_df, columns["source_columns_to_embed"], columns["hierarchical_selection_column_name"])
                columns["source_columns_to_embed"] = [columns["hierarchical_selection_column_name"]]
            else:
                print("No hierarchical selection implemented. Please provide a column name if you want to perform hierarchical selection.")

            source_df = translate_and_embed(source_df, columns["source_columns_to_translate"], 
                                            columns["source_columns_to_embed"], paths["source_translated_file"], 
                                            paths["source_embedded_file"], os.environ["OPENAI_API_KEY"],
                                            export_files=export_files)
        else:
            source_df = pd.read_hdf(paths["source_embedded_file"], key='df')
            if automated_column_labeling:
                columns = assign_columns(os.environ["OPENAI_API_KEY"],columns,source_df.drop(columns=columns.get("source_confidential_column", []), errors='ignore'))

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
        matched_df = match_datasets(source_df, target_df, gpt_model="gpt-4o-mini", api_key=os.environ["OPENAI_API_KEY"], top_n=5, chunk_size=1500)
        if matched_output_export:
            matched_df.drop(columns=["embedding"]).to_excel(paths["output_file"], index=False)

        output_columns = list(source_df.columns) + columns["target_columns_to_keep"] 

        df_converted = matched_df.copy()
        if currency_settings and os.environ["FRED_API_KEY"]:
            converted_currency_serie, correct_inflation_serie = currency_converter(matched_df, 
                                            target_currency=currency_settings["target_currency"], 
                                            date_column=columns["date_column"], 
                                            currency_column=columns["unit_column"], 
                                            amount_column=columns["amount_column"], 
                                            target_inflation_year=currency_settings["target_inflation_year"], 
                                            fred_api_key=os.environ["FRED_API_KEY"])
            converted_currency_name = f'Amount in {currency_settings["target_currency"]}'
            correct_inflation_name = f'Amount in {currency_settings["target_currency"]} corrected inflation in {currency_settings["target_inflation_year"]}'
            df_converted[converted_currency_name] = converted_currency_serie
            df_converted[correct_inflation_name] = correct_inflation_serie
            df_converted["CO2e"] = df_converted[f'Amount in {currency_settings["target_currency"]} corrected inflation in {currency_settings["target_inflation_year"]}'] * df_converted[columns["emission_factor_column"]]
            output_columns = output_columns + [converted_currency_name, correct_inflation_name]

            if currency_settings["target_currency_additional"]:
                converted_CHF_currency_serie,_ = currency_converter(matched_df, 
                                                target_currency=currency_settings["target_currency_additional"], 
                                                date_column=columns["date_column"], 
                                                currency_column=columns["unit_column"], 
                                                amount_column=columns["amount_column"], 
                                                target_inflation_year=currency_settings["target_inflation_year"], 
                                                fred_api_key=os.environ["FRED_API_KEY"])
                additional_currency_name = f'Amount in {currency_settings["target_currency_additional"]}'
                df_converted[additional_currency_name] = converted_CHF_currency_serie
                output_columns.append(additional_currency_name)
        elif currency_settings and not os.environ["FRED_API_KEY"]:
            Warning('''No currency conversion performed. Please provide a target currency and a FRED API key to convert the currency. Check the config file.\n
                    We consider that the amount in the target currency is already in the target currency and that the inflation is already corrected.''') 
        else:
            df_converted["CO2e"] = df_converted[columns["amount_column"]] * df_converted[columns["emission_factor_column"]]
        
        # Calculate the total uncertainty factor for each row
        if (columns["target_columns_uncertainty"]) and (columns["target_columns_uncertainty_80pct"]):
            df_converted['total_uncertainty_factor'] = df_converted[columns["target_columns_uncertainty"]] + \
                                            (0.8 * df_converted[columns["target_columns_uncertainty_80pct"]])

            # Calculate the uncertainty in carbon emissions for each row
            df_converted['uncertainty_carbon_emissions'] = df_converted['CO2e'] * df_converted['total_uncertainty_factor']
            output_columns = output_columns + [columns["target_columns_uncertainty"], columns["target_columns_uncertainty_80pct"]]

        output_columns = output_columns + ['combined_target', 'CO2e'] 

        final_df = df_converted[output_columns].drop(columns=['embedding'])

        final_df = final_df.loc[:,~final_df.columns.duplicated()]
        
        if semantic_error_estimation: 
            logging.info("Estimating semantic errors")
            flagged_df = find_semantic_mismatches_batch(final_df, batch_size=200)
            logging.info(f"output file saved at {paths['output_file']}")
            output_hyper_path = get_file_paths(paths["output_file"], ".hyper")
            df_to_hyper(flagged_df, output_hyper_path)
            final_df = flagged_df.copy()

        final_df.to_excel(paths["output_file"], index=False)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise

if __name__ == '__main__':    
    from preprocess_physical_database import *
    env_path = os.path.expanduser('~/global_env/.env')
    load_dotenv(env_path)
    os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI')
    chat_bot_structured_input()
    parser = argparse.ArgumentParser(description="Run the main script with config path.")
    
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    parser.add_argument('--reset', type=bool, required=False, help="Reset the output files, please close them before running the script")
    parser.add_argument('--semantic_error_estimation', type=bool, required=False, help="Export only the matched output without carbon calculations")
    parser.add_argument('--matched_output_export', type=bool, required=False, help="semantic_error_estimation to estimate the errors")
    #args = parser.parse_args()
    main(config_path="scripts/configs/config.yaml", reset=False, semantic_error_estimation=True, matched_output_export=True)
