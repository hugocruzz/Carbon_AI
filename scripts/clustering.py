# Import libraries
import requests
import numpy as np
import pandas as pd
import sys
from sentence_transformers import SentenceTransformer
from more_itertools import chunked
from sklearn.cluster import MiniBatchKMeans
import pandas as pd
from functions import check_and_normalize_series
from openai import OpenAI
import json
import os 
from dotenv import load_dotenv

def clustering(df, embedding_column_name="combined", combined_column_name="combined", embedding_output_path ="data\output\embeddings_mpnet_temp.pkl", n_clusters=12, batch_size=64):
    # Load data
    df[embedding_column_name] = df[embedding_column_name].fillna("")
    df[combined_column_name] = df[embedding_column_name].apply(lambda x: ' '.join(x.astype(str)), axis=1)
    df_unique = df[[combined_column_name]].drop_duplicates()
    df_unique[combined_column_name] = check_and_normalize_series(df_unique[combined_column_name])
    
    normalized_unique_df = perform_clustering(df_unique, embedding_output_path, batch_size, n_clusters = n_clusters)
    
    df_final = df.merge(normalized_unique_df, left_on=combined_column_name, right_on=combined_column_name, how="left")
    return df_final

def perform_clustering(df, embedding_output_path, batch_size, n_representatives = 10, n_clusters = 12):

    env_path = os.path.expanduser('~/global_env/.env')
    load_dotenv(env_path)
    if not os.getenv('OPENAI'):
        raise ValueError('OPENAI key is not set in the environment variables')
    else:
        os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI')
    df["text"] = df["combined"].copy()
    # Load model
    if not os.path.exists(embedding_output_path):
        model = SentenceTransformer('all-mpnet-base-v2', device='cpu')

        embeddings = []
        for batch_text in chunked(df['text'].tolist(), batch_size):
            embeddings.extend(model.encode(batch_text, show_progress_bar=False))
            sys.stdout.write(f'\rprocessed for {len(embeddings)} works so far... (out of {df.shape[0]})')
            sys.stdout.flush()

        embeddings = pd.DataFrame(embeddings, index=df.index)

        embeddings.to_pickle(embedding_output_path)
    else:
        embeddings = pd.read_pickle(embedding_output_path)

    kmeans = MiniBatchKMeans(n_clusters=n_clusters, n_init='auto')
    batch_size = 10000
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings.iloc[i:i+batch_size]
        kmeans.partial_fit(batch)

    df['cluster'] = kmeans.predict(embeddings)
    representatives = {}

    for cluster in df['cluster'].unique():
        cluster_embeddings = embeddings[df['cluster'] == cluster]

        # If there are fewer samples than n_representatives, use all samples
        if len(cluster_embeddings) < n_representatives:
            global_indices = cluster_embeddings.index
        else:
            # Use Kmeans to select n_representatives
            subcluster = MiniBatchKMeans(n_clusters=n_representatives, n_init='auto')
            subcluster.fit(cluster_embeddings)

            # Get distance of each point to each cluster
            distances = subcluster.transform(cluster_embeddings)

            # Retrieve indices of samples closest to centroids
            closest_indices = np.argmin(distances, axis=0)
            global_indices = cluster_embeddings.iloc[closest_indices].index

        # Align with global indices and collect the representative texts
        representatives[int(cluster)] = df.loc[global_indices, 'text'].str.slice(0, 400).tolist()

    prompt = ''
    prompt += f'I have {n_clusters} clusters of different purchases of a technical university. For each clusters, I give you a description and a categorization that are part of the cluster.\n'
    prompt += f'For each cluster, find a short title (3-5 words), and a description (2-3 sentences) that best describe the content of the clusters. Here are the clusters: \n'
    for cluster_id, content in representatives.items():
        prompt += f'\n\n'
        prompt += f'CLUSTER: {cluster_id}\n'
        for e in content:
            prompt += f' - {e}\n'
    prompt += "\n Provide the data in JSON with the following format: [{'id':1, 'title':'Lorem Ipsum', 'description':'Lorem Ipsum...'}, ...]. Do not wrap the json codes in JSON markers"


    # Important: use the 'Secrets' feature from Google Colab (icon of the key) and set a variable named OPENAI_API_KEY with your API key from OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"] )


    if not os.path.exists('cluster_description.pkl'):
        response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": prompt
                }
            ]
            }
        ],
        temperature=1,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        )
        answer_gpt = response.choices[0].message.content
        
        answer_gpt = answer_gpt.replace("'", '"')
        print (answer_gpt)
        cluster_description = pd.DataFrame(json.loads(answer_gpt)).add_prefix("cluster_")

        cluster_description.to_pickle('data/output/cluster_description.pkl')
    else:
        cluster_description = pd.read_pickle('cluster_description.pkl')

    # Join the description of the clusters obtained with ChatGPT to the work downloaded from OpenAlex
    df = df.set_index('cluster').join(cluster_description.set_index('cluster_id')).reset_index()
    return df