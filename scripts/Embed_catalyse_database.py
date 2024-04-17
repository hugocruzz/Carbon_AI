import openai
import pandas as pd
from openai.embeddings_utils import get_embedding
import os 

# Load the DataFrame
df = pd.read_excel(r'data\achats_EPFL\Liste produits CATALYSE SV 2022.xlsx')
df = df[df["Libellé FR Centre de coût"]=="Unité de la Prof. Gisou van der Goot"]
#Drop column "Créé par Nom" and "Créé par Prénom"
df = df.drop(columns=["Créé par Nom", "Créé par Prénom"])

embedding_column_name = ["Désignation article", "Famille_libellé anglais", "Fournisseur"]
output_file_name = 'data/Results/achats_GVG_lab_2022.pkl'

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
