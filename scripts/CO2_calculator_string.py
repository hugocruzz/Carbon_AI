import openai
import os
import pandas as pd
import tiktoken
import numpy as np

from openai.embeddings_utils import get_embedding, cosine_similarity

def search_in_df(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002",
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))
    results_score = df["similarity"].max()
    results_index = df["similarity"].argmax()
    return (results_index, results_score)

input_string = "Ordinateur de bureau"
with open(r"C:\Users\cruz" + r'\API_openAI.txt', 'r') as f:
    read_api_key = f.readline()
    openai.api_key = read_api_key

# %%
def __main__(input_string, API_key):
    os.environ["OPENAI_API_KEY"] = read_api_key
    path_database =r"data\processed\nacres\NACRES_with_embeddings_and_factors.pkl"

    df = pd.read_pickle(path_database)
    search_output = search_in_df(df, input_string, n=1, pprint=False)
    df_object = df.iloc[search_output[0]]
    # Perform the calculation using the "FEL1P5" field from the df_object and the input price
    calculation_result = df_object["FEL1P5"] * input_price
    similarity_score = int(search_output[1]*100)/100
    result = (f"'{input_string}' is categorized as '{df_object['Intitulés Nacres']}' "
                f"(Classe: {df_object['Classe']}). \nThe similarity score is {similarity_score}. \n"
                f"Calculation Result: {calculation_result} kg CO2e.")
if __name__ == '__main__':
    __main__(input_string, read_api_key)