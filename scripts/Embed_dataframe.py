import pandas as pd
from openai import OpenAI
import os 
from functions import check_and_normalize_series
def embed_dataframe(df, embedding_column_name, combined_column_name="combined", output_embedding_name="embedding", api_key=None, embedding_model="text-embedding-ada-002"):
    """Embed dataframe columns using OpenAI embeddings."""
    
    client = OpenAI(api_key=api_key)

    def get_embeddings(texts, model="text-embedding-ada-002"):
        texts = [text.replace("\n", " ") for text in texts]
        response = client.embeddings.create(input=texts, model=model)
        return [data.embedding for data in response.data]

    def batch_embeddings(texts, batch_size=1500, export_chunk=False):
        total_batches = (len(texts) + batch_size - 1) // batch_size
        embeddings = []

        for i in range(0, len(texts), batch_size):
            current_batch = (i // batch_size) + 1
            print(f"Processing batch {current_batch} of {total_batches}...")
            batch = texts[i:i+batch_size]
            batch_embeddings = get_embeddings(batch, model=embedding_model)
            if export_chunk:
                pd.DataFrame(batch_embeddings).to_csv(f"data/results/embeddings_{current_batch}.csv")

            embeddings.extend(batch_embeddings)
        
        print("All batches processed.")
        return embeddings
    df[embedding_column_name] = df[embedding_column_name].fillna("")
    df[combined_column_name] = df[embedding_column_name].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    df_unique = df[[combined_column_name]].drop_duplicates()
    normalized_unique_df = check_and_normalize_series(df_unique[combined_column_name])
    df_unique["embedding"] = batch_embeddings(normalized_unique_df, batch_size=1500, export_chunk=False)
    #Merge df with embedding based on index
    df_final = df.merge(df_unique, left_on=combined_column_name, right_on=combined_column_name, how="left")
    return df_final

if __name__ == '__main__':
    path = r'data\Results\Test_articles_translated.xlsx'
    output_file_name = 'data/Results/test_dataset.pkl'
    df = pd.read_excel(path)
    if "Créé par Nom" in df.columns:
        df.drop(columns=["Créé par Nom", "Créé par Prénom"], inplace=True)
        
    df = embed_dataframe(df, ["Désignation article", "Famille"], combined_column_name="combined", output_embedding_name="embedding")
    df.to_pickle(output_file_name)
