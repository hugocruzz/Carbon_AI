from tableauhyperapi import HyperProcess, Connection, Telemetry, TableDefinition, SqlType, Inserter, TableName, CreateMode
import pandas as pd
import logging
import json
import yaml
import os 
from dotenv import load_dotenv
from openai import OpenAI

def find_columns_labels(source_df, api_key=None, description_col_nb=2, model = "gpt-4o-mini"):
    df_head = source_df.head().to_json(orient='records')  
    Category_name_possibilities = ["Description", "Sub_category", "Category", "Sub_family", "Family", "sub_domain", "domain"]   
    example_json = {"date_column": "Date", "amount_column": "Amount", "unit_column": "Currency", "description_column": Category_name_possibilities[:description_col_nb]}
    prompt = f"""
    You are given a DataFrame with the following head: {df_head}. Your task is to identify and provide the column names that best represent the following information:

    1. **Date of Purchase**: The column that represents the date on which the purchase was made.
    2. **Amount**: The column that specifies the amount (money for example) involved in the purchase.
    3. **Unit**: The column that indicates the unit of the article purchased.
    4. **Description**: Columns that describe the article in a way understandable by a human, the values in these column should be a text description, not a number.

    Please consider both the column labels and the values they contain to ensure accurate identification. From the DataFrame, select {description_col_nb} columns that collectively provide the most detailed description of the purchase. Assign them as:

    - "date_column"
    - "amount_column"
    - "unit_column"
    - "description_column"

    Provide your response in a valid JSON format following this schema: {json.dumps(example_json)}
    """

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    success = False

    while not success:
        chat_completion = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": json.dumps(df_head)}
                    ]
                )
        
        data = chat_completion.choices[0].message.content
        data_json = json.loads(data)
        success = True
    #Change key name of "description_column" to "source_columns_to_embed"
    data_json["source_columns_to_embed"] = data_json.pop("description_column")

    return data_json

def load_global_env():
    env_path = os.path.expanduser('~/global_env/.env')
    if os.path.exists(env_path):
        load_dotenv(env_path)
    else:
        raise FileNotFoundError(f"No global .env file found at {env_path}")

def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_file_paths(base_path: str, suffix: str) -> str:
    """Generate file paths with given suffix."""
    return base_path.replace("." + base_path.split(".")[-1], suffix)
def emphasize_and_combine_columns(df: pd.DataFrame, source_columns_emphasis: list, source_columns_to_embed: list) -> pd.DataFrame:
    """
    Emphasizes the specified columns by adding asterisks around non-empty strings, 
    replaces "**" with an empty string, and combines the specified columns into one.

    Args:
    df (pd.DataFrame): The input DataFrame.
    source_columns_emphasis (list): The columns to emphasize.
    source_columns_to_embed (list): The columns to combine into a single column.

    Returns:
    pd.DataFrame: The modified DataFrame with emphasized and combined columns.
    """
    df[source_columns_emphasis] = df[source_columns_emphasis].fillna("")
    df[source_columns_emphasis] = df[source_columns_emphasis].apply(lambda x: "*" + x + "*")
    df[source_columns_emphasis] = df[source_columns_emphasis].replace(r'^\*{2}$', '', regex=True)
    df["combined"] = df[source_columns_to_embed].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    return df

def hierarchical_selection(source_df,source_columns_to_embed, merged_column, reverse_col=False):
    source_df_copy = source_df.copy()
    source_columns_to_embed = source_columns_to_embed[::-1]
    source_df_copy[merged_column] = source_df_copy[source_columns_to_embed[0]]
    for col in source_columns_to_embed[1:]:
        source_df_copy[merged_column] = source_df_copy[merged_column].combine_first(source_df_copy[col])

    return source_df_copy

def pre_process_source_df(source_columns_to_embed, source_df):

    source_df = source_df.dropna(subset=source_columns_to_embed,how = 'all')
    # Remove numbers from the specified columns
    for column in source_columns_to_embed:
        source_df[column] = source_df[column].str.replace(r'\d+', '', regex=True)
    return source_df

def update_dataframe_with_correction(source_df, corrected_df, key_column):
    '''Update the source DataFrame with corrected values from the corrected DataFrame coming from a manual correction.'''
    source_df_copy = source_df.copy()
    corrected_df_copy = corrected_df.copy()
    
    # Reset index to ensure no duplicate labels interfere with the merge
    source_df_copy.reset_index(drop=True, inplace=True)
    corrected_df_copy.reset_index(drop=True, inplace=True)
    
    # Merging dataframes on the key_column
    merged_df = pd.merge(source_df_copy, corrected_df_copy, on=key_column, how='left', suffixes=('', '_corrected'))
    
    # List of columns to update
    columns_to_update = [col for col in corrected_df_copy.columns if col != key_column]
    
    # Update the columns in source_df_copy with values from corrected_df
    for col in columns_to_update:
        source_df_copy[col] = merged_df.apply(
            lambda row: row[f'{col}_corrected'] if pd.notnull(row[f'{col}_corrected']) else row[col], axis=1
        )
    
    return source_df_copy
    

def get_sqltype(dtype):
    """Convert pandas dtype to Tableau Hyper SQLType."""
    if pd.api.types.is_integer_dtype(dtype):
        return SqlType.big_int()
    elif pd.api.types.is_float_dtype(dtype):
        return SqlType.double()
    elif pd.api.types.is_datetime64_any_dtype(dtype):
        return SqlType.timestamp()
    elif pd.api.types.is_bool_dtype(dtype):
        return SqlType.bool()
    else:
        return SqlType.text()

def df_to_hyper(df, output_path):
    """Export a pandas DataFrame to a Tableau Hyper file."""
    
    # Ensure all data is in the correct format before insertion
    receiver_data = []
    for _, row in df.iterrows():
        row_data = []
        for value in row:
            if pd.isnull(value):  # Handle NaN values
                row_data.append(None)
            else:
                row_data.append(value)
        receiver_data.append(row_data)
    
    # Define the table schema dynamically based on DataFrame columns
    table_definition = TableDefinition(
        table_name=TableName("Extract", "Extract"),
        columns=[
            TableDefinition.Column(col, get_sqltype(dtype)) for col, dtype in zip(df.columns, df.dtypes)
        ]
    )

    # Start the Hyper process and create the Hyper file
    with HyperProcess(telemetry=Telemetry.SEND_USAGE_DATA_TO_TABLEAU) as hyper:
        with Connection(endpoint=hyper.endpoint, database=output_path, create_mode=CreateMode.CREATE_AND_REPLACE) as connection:
            connection.catalog.create_schema("Extract")
            connection.catalog.create_table(table_definition)
            with Inserter(connection, table_definition) as inserter:
                inserter.add_rows(receiver_data)
                inserter.execute()
                
def load_api_key(api_key_path: str) -> str:
    """Load API key from a file."""
    with open(api_key_path, 'r') as f:
        return f.readline().strip()