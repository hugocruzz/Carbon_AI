import openai
import pandas as pd
from openai import OpenAI
import os 

# Load the DataFrame
df = pd.read_excel(r'data\Results\Test_articles_translated.xlsx')
#df = df[df["Libellé FR Centre de coût"]=="Unité de la Prof. Gisou van der Goot"]
if "Créé par Nom" in df.columns:
    df.drop(columns=["Créé par Nom", "Créé par Prénom"], inplace=True)

'''
df_unspsc = pd.read_excel(r"data\Raw\Tableau UNSPSC V10 Full_Raw_1 (006).xlsx")

#Take the first 2 digits of the UNSPSC code and add "0000000"
df_copy = df.copy()
df_copy = df_copy[~df_copy["Famille1"].isna()]
df_copy["Famille1"] = df_copy["Famille1"].astype(int).astype(str)
df_copy["Segment_code"] = df_copy["Famille1"].str[:2] + "000000"
#Merge the two dataframes but add only the column "English Name"
df_copy.rename(columns={"English Name": "UNSPSC_Segment"}, inplace=True)
df_copy[df_copy["Segment_code"]!=df_copy["Famille1"]]
df_unspsc["Code"] = df_unspsc["Code"].astype(str)
df_copy = df_copy.merge(df_unspsc[["Code", "English Name"]], left_on="Segment_code", right_on="Code", how="left")
#Rename the column "English Name" to "UNSPSC_Segment"

#Filter all the elements where df_copy["Code"]==df_copy["Segment_code"]
df_copy["Family_code"] = df_copy["Famille1"].astype(str).str[:4] + "0000"
#Filter all the rows that ends with "0000"
df_unspsc = df_unspsc[df_unspsc["Code"].str.endswith("0000")]
'''
#embedding_column_name = ["Désignation article", "Famille_libellé anglais", "Fournisseur"]
embedding_column_name = ["Désignation article", "Famille"]
output_file_name = 'data/Results/test_dataset.pkl'

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
