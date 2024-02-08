from flask import Flask, request, jsonify
import os
import pandas as pd
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

app = Flask(__name__)

# Load products DataFrame
nacres_database_path = "data/NACRES_with_embeddings_and_factors.pkl"
nacres_df = pd.read_pickle(nacres_database_path)

agribalyse_database_path = "data/agribalyse_embeddings_factors.pkl"
agribalyse_df = pd.read_pickle(agribalyse_database_path)

# Function to search in DataFrame
def search_in_products(products_df, query_description, top_n=3):
    query_embedding = get_embedding(
        query_description,
        engine="text-embedding-ada-002",
    )
    products_df["similarity"] = products_df["embedding"].apply(lambda x: cosine_similarity(x, query_embedding))
    highest_similarity_score = products_df["similarity"].max()
    best_match_index = products_df["similarity"].argmax()
    return best_match_index, highest_similarity_score

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    product_description = data.get('product_description')
    budget = data.get('amount')
    api_key = data.get('api_key')
    database_name = data.get('database_name')

    if api_key is None:
        api_key = os.environ.get('OPENAI_API_KEY')
    if database_name is None:
        database_name = "labo1point5"

    # Set API key
    openai.api_key = api_key
    if database_name=="labo1point5":
        products_df = nacres_df
        category_database_name = "Intitulés Nacres"
        code_database_name = "CodeNACRES"
        emission_factor_name = "FEL1P5"
        emission_factor_name_unit = "kg CO2e/euros dépensés"
    elif database_name=="agribalyse":
        products_df = agribalyse_df
        category_database_name = "Nom du Produit en Français"
        code_database_name = "Code\nAGB"
        emission_factor_name = "kg CO2 eq/kg de produit"
        emission_factor_name_unit = "kg CO2e/kg de produit"

    # Search in DataFrame
    best_match = search_in_products(products_df, product_description, top_n=1)
    matched_product = products_df.iloc[best_match[0]]
    
    # Calculate similarity and emissions
    similarity = round(best_match[1] * 100) / 100
    emission_factor = matched_product[emission_factor_name]
    CO2_emitted = emission_factor * budget if budget is not None else None
    # Prepare the result
    response_data = {
        'matched_category': matched_product[category_database_name],
        'matched_code': matched_product[code_database_name],
        'similarity_score': similarity,
        'CO2_emitted': CO2_emitted,
        'CO2_unit': 'kg CO2e',
        'emission_factor': emission_factor,
        'emission_factor_unit': emission_factor_name_unit
    }

    return jsonify(response_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use port 5000 if not specified
    app.run(host="0.0.0.0", port=port)
