import openai
import os
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

def get_embedding(text, client, model="text-embedding-ada-002"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def search_in_df(df, product_description, n=3, pprint=True):
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    product_embedding = get_embedding(
        product_description,client
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(np.array(x).reshape(1,-1), np.array(product_embedding).reshape(1,-1)))
    results_score = df["similarity"].max()
    results_index = df["similarity"].argmax()
    return (results_index, results_score)

input_string = "Ordinateur de bureau"
with open(r"C:\Users\cruz" + r'\API_openAI.txt', 'r') as f:
    read_api_key = f.readline()
    openai.api_key = read_api_key

# %%
def __main__(input_string, API_key, factor_type="Monetary"):
    os.environ["OPENAI_API_KEY"] = read_api_key
    path_database =r"data\NACRES_with_embeddings_and_factors.pkl"

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