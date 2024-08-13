import numpy as np
import pandas as pd
import logging
from translate_db import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


if __name__ == '__main__':
        
    target_df = pd.read_excel(r"data\Raw\Target_PER1p5_nacres_fe_database_v1-0-2023.xlsx", sheet_name="NACRES-EF")
    target_df["nacres.description.en"] = target_df["nacres.description.en"].str.rstrip()
    #OPen data\Raw\nacres-janvier-2022.xls
    nacres_df = pd.read_excel(r"data\Raw\nacres-janvier-2022.xls", sheet_name="M9 NACRES")
    #rename columns nacres_df["Libellé étendu nouvelle nomenclature achat"] to "nacres.description.en" and "Codes Nacres" to "nacres.code"
    nacres_df.rename(columns={"Codes Nacres": "nacres.code"}, inplace=True)
    nacres_df.rename(columns={"Libellé étendu nouvelle nomenclature achat": "nacres.description.en"}, inplace=True)

    #For each nacres_df["Codes Nacres"] that has a length of less than 5, append these rows to target_df
    nacres_df["nacres.code"] = nacres_df["nacres.code"].astype(str)
    nacres_df["nacres.code"] = nacres_df["nacres.code"].str.strip()
    nacres_df["nacres.code"] = nacres_df["nacres.code"].str.replace(".", "")
    nacres_df["nacres.description.en"] = nacres_df["nacres.description.en"].str.rstrip()

    #append rows to target_df, without using append
    nacres_df_domain = nacres_df[nacres_df["nacres.code"].str.len() < 4]
    nacres_df_domain = nacres_df_domain[['nacres.code', 'nacres.description.en']]
    nacres_df_domain_translated = translate_db(nacres_df_domain, ['nacres.description.en'])
    #rename nacres.description.en original to nacres.description.fr
    nacres_df_domain_translated.rename(columns={"nacres.description.en original": "nacres.description.fr"}, inplace=True)
    target_df_2 = pd.concat([target_df, nacres_df_domain_translated], ignore_index=True)

    def get_median_for_subcategories(df, code, column):
        subcategory_mask = df['nacres.code'].str.startswith(code) & (df['nacres.code'].str.len() > len(code))
        subcategory_values = df.loc[subcategory_mask, column].dropna().values  # Ensure to drop NaNs
        if len(subcategory_values) > 0:
            median_value = np.nanmedian(subcategory_values)
        else:
            median_value = np.nan  # Define a default value when no subcategory values are found
        return median_value

    # Function to replace NaNs with medians in multiple float columns
    def replace_nan_with_median(df):
        float_columns = df.select_dtypes(include=['float64']).columns

        for column in float_columns:
            for i, row in df.iterrows():
                code = row['nacres.code']
                if len(code) < 5 and pd.isna(row[column]):
                    median_value = get_median_for_subcategories(df, code, column)
                    if not pd.isna(median_value):  # Check if median_value is not NaN before replacing
                        df.at[i, column] = median_value

        return df

    # Replace NaNs with the calculated medians
    df = replace_nan_with_median(target_df_2)
    df.to_excel(r"data\Raw\Target_pre_processed_PER1p5_nacres_fe_database_v1-0-2023.xlsx", index=False)
