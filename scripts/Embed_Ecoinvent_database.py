import openai
import pandas as pd
from openai.embeddings_utils import get_embedding
import os 

# Load the DataFrame
df = pd.read_excel(r'data\Raw\Cut-off Cumulative LCIA v3.10.xlsx', sheet_name="LCIA", header=3)
#Get index of unique values of Reference product Name
df = df.groupby("Reference Product Name").first().reset_index()

'''
# read data\Raw\countries-codes.parquet 
iso_country_codes = pd.read_parquet(r'data\Raw\countries-codes.parquet')[["iso2_code","label_en"]]

#Preprocess ecoinvent
new_rows = pd.DataFrame([
    {"iso2_code": "RoW", "label_en": "Rest of the World"},
    {"iso2_code": "GLO", "label_en": "Global"},
    {"iso2_code": "RER","label_en":"Europe"},
])
iso_country_codes = iso_country_codes._append(new_rows, ignore_index=True)
df["Geography name"] = df["Geography"].map(iso_country_codes.set_index('iso2_code')['label_en'])
df["Geography name"] = df["Geography name"].fillna(df["Geography"])
'''

embedding_column_name = "Activity Name"
output_file_name = 'data/Ecoinvent_cut_off_cumulative_embeddings_factors.pkl'

# Embedding model parameters
embedding_model = "text-embedding-ada-002"
embedding_encoding = "cl100k_base"  # Encoding for text-embedding-ada-002
max_tokens = 8000  # Max tokens for ada-002
with open(r"C:\Users\cruz" + r'\API_openAI.txt', 'r') as f:
    read_api_key = f.readline()
    openai.api_key = read_api_key

os.environ["OPENAI_API_KEY"] = read_api_key
def batch_embeddings(texts, batch_size=1500):
    total_batches = (len(texts) + batch_size - 1) // batch_size  # Calculate the total number of batches
    embeddings = []

    for i in range(0, len(texts), batch_size):
        current_batch = (i // batch_size) + 1  # Calculate the current batch number
        print(f"Processing batch {current_batch} of {total_batches}...")  # Print the progress
        batch = texts[i:i+batch_size]
        batch_embeddings = [get_embedding(text, engine=embedding_model) for text in batch]
        embeddings.extend(batch_embeddings)
    
    print("All batches processed.")
    return embeddings
df["combined"] = df[embedding_column_name]

# Apply batch embedding process
df['embedding'] = batch_embeddings(df["combined"], batch_size=1500)

# Save the DataFrame with embeddings
df.to_pickle(output_file_name)
