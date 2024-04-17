import openai
import pandas as pd
from openai.embeddings_utils import get_embedding
import os 

# Load the DataFrame
df = pd.read_excel('data\Raw\PER1p5_nacres_fe_database_v1-0-2023.xlsx', sheet_name="NACRES-EF")
embedding_column_name = "nacres.description.en"
output_file_name = 'data/achats_EPFL/NACRES_embedded.pkl'

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
        # Assuming get_embedding is a function that processes each batch
        # Replace the following line with your actual function call to get embeddings
        batch_embeddings = [get_embedding(text, engine=embedding_model) for text in batch]
        embeddings.extend(batch_embeddings)
    
    print("All batches processed.")
    return embeddings


# Apply batch embedding process
df['embedding'] = batch_embeddings(df[embedding_column_name].tolist(), batch_size=1500)

# Save the DataFrame with embeddings
df.to_pickle(output_file_name)
