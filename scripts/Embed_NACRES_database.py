import openai
import pandas as pd
import os 
from openai import OpenAI



# Load the DataFrame
df = pd.read_excel('data\Raw\PER1p5_nacres_fe_database_v1-0-2023.xlsx', sheet_name="NACRES-EF")
df["category"] = df["category"].str.replace(".", " ")
embedding_column_name = ["nacres.description.en", "ademe.description", "useeio.description", "module", "category"]
output_file_name = 'data/achats_EPFL/NACRES_embedded.pkl'

# Embedding model parameters
embedding_model = "text-embedding-ada-002"

with open(r"C:\Users\cruz" + r'\API_openAI.txt', 'r') as f:
    read_api_key = f.readline()
    openai.api_key = read_api_key

client = OpenAI(api_key=read_api_key)

def get_embedding(text, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def batch_embeddings(texts, batch_size=1500):
    total_batches = (len(texts) + batch_size - 1) // batch_size  # Calculate the total number of batches
    embeddings = []

    for i in range(0, len(texts), batch_size):
        current_batch = (i // batch_size) + 1  # Calculate the current batch number
        print(f"Processing batch {current_batch} of {total_batches}...")  # Print the progress
        batch = texts[i:i+batch_size]
        # Assuming get_embedding is a function that processes each batch
        # Replace the following line with your actual function call to get embeddings
        batch_embeddings = [get_embedding(text, model=embedding_model) for text in batch]
        embeddings.extend(batch_embeddings)
    
    print("All batches processed.")
    return embeddings


#Take only unique df[embedding_column_name] values, create a dataframe
#Create a combined column of all the embedding_column_name values, embedding_column_name can have a len of 1 or more
df["combined"] = df[embedding_column_name].apply(lambda x: ' '.join(x.astype(str)), axis=1)
df_unique = df["combined"].unique()
#Convert it to dataframe with column name = embedding_column_name
df_unique = pd.DataFrame(df_unique, columns=["combined"])
import time
tik = time.time()
df_unique['embedding'] = batch_embeddings(df_unique["combined"], batch_size=1500)
tok = time.time()
print(f"Time taken to process embeddings: {tok - tik} seconds")
#Merge df_unique to df so we get back our original dataframe
df = df.merge(df_unique, left_on="combined", right_on="combined")

# Save the DataFrame with embeddings
df.to_pickle(output_file_name)
