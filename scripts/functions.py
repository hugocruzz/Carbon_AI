from tableauhyperapi import HyperProcess, Connection, Telemetry, TableDefinition, SqlType, Inserter, TableName, CreateMode
import pandas as pd

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