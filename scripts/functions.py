from tableauhyperapi import HyperProcess, Connection, Telemetry, TableDefinition, SqlType, Inserter, TableName, CreateMode
import pandas as pd

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
