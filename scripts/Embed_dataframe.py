import openai
import pandas as pd
from openai import OpenAI
import os 


def embed_dataframe(df, embedding_column_name, combined_column_name="combined", output_embedding_name="embedding", api_key=None, output_file_name=None):
    # Embedding model parameters
    embedding_model = "text-embedding-ada-002"

    client = OpenAI(api_key=api_key)

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
    df[combined_column_name] = df[embedding_column_name].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    df_unique = df[combined_column_name].unique()
    #Convert it to dataframe with column name = embedding_column_name
    df_unique = pd.DataFrame(df_unique, columns=[combined_column_name])
    df_unique[output_embedding_name] = batch_embeddings(df_unique[combined_column_name], batch_size=1500)
    #Merge df_unique to df so we get back our original dataframe
    df = df.merge(df_unique, left_on=combined_column_name, right_on=combined_column_name)
    if output_file_name:
        # Save the DataFrame with embeddings
        df.to_pickle(output_file_name)
    return df

if __name__ == '__main__':
    path = r'data\Results\Test_articles_translated.xlsx'
    output_file_name = 'data/Results/test_dataset.pkl'
    # Load the DataFrame
    df = pd.read_excel(path)

    if "Créé par Nom" in df.columns:
        df.drop(columns=["Créé par Nom", "Créé par Prénom"], inplace=True)
        
    df = embed_dataframe(df, ["Désignation article", "Famille"], combined_column_name="combined", output_embedding_name="embedding")
    # Save the DataFrame with embeddings
    df.to_pickle(output_file_name)


