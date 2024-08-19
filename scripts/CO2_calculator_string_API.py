from flask import Flask, request, jsonify
import os
import pandas as pd
import openai

app = Flask(__name__)

# Load DataFrames
data_paths = {
    "nacres": "data/NACRES_with_embeddings_and_factors.pkl",
    "agribalyse": "data/agribalyse_embeddings_factors.pkl",
    "mobitool": "data/mobitool_embeddings_factors.pkl",
    "unspsc": "data/unspsc_ada_embeddings.pkl"
}
data_frames = {name: pd.read_pickle(path) for name, path in data_paths.items()}

# Configuration for each database
database_configs = {
    "labo1point5": {
        "df": "nacres",
        "category_name": "Intitulés Nacres",
        "code_name": "CodeNACRES",
        "emission_factor_name": "FEL1P5",
        "emission_factor_unit": "kg CO2e",
        "functional_unit_name": "unite fonctionnelle"
    },
    "agribalyse": {
        "df": "agribalyse",
        "category_name": "Nom du Produit en Français",
        "code_name": "Code\nAGB",
        "emission_factor_name": "kg CO2 eq/kg de produit",
        "emission_factor_unit": "kg CO2e",
        "functional_unit_name": "unite fonctionnelle"
    },
    "mobitool": {
        "df": "mobitool",
        "category_name": "Véhicule",
        "code_name": "combined",
        "emission_factor_name": "somme [g CO2-éq.]",
        "emission_factor_unit": "g CO2e",
        "functional_unit_name": "unite fonctionnelle"
    },
    "unspsc": {
        "df": "unspsc",
        "category_name": "English Name",
        "code_name": "Code",
        "emission_factor_name": None,
        "emission_factor_unit": None,
        "functional_unit_name": None
    }
}

def get_query_embedding(query_description):
    return get_embedding(query_description, engine="text-embedding-ada-002")

def find_top_matches(products_df, query_embedding, top_n=3):
    products_df["similarity"] = products_df["embedding"].apply(lambda x: cosine_similarity(x, query_embedding))
    return products_df.sort_values(by="similarity", ascending=False).head(top_n)

def format_response(matched_product, config, similarity, amount=None):
    functional_unit = matched_product[config["functional_unit_name"]] if config["functional_unit_name"] else None
    emission_factor_name = matched_product[config["emission_factor_name"]] if config["emission_factor_name"] else None
    emission_factor_unit = config["emission_factor_unit"]
    CO2_emitted = matched_product[config["emission_factor_name"]] * amount if amount else None

    
    return {
        'matched_category': matched_product[config["category_name"]],
        'matched_code': matched_product[config["code_name"]],
        'similarity_score': round(similarity * 100) / 100,
        'CO2_emitted': CO2_emitted,
        'emission_factor': emission_factor_name,
        'emission_factor_unit': emission_factor_unit,
        'functional_unit': functional_unit
    }

@app.route('/search', methods=['POST'])
def search():
    data = request.json
    product_description = data.get('product_description', '')
    amount = data.get('amount')
    database_name = data.get('database_name', 'labo1point5')
    top_n = data.get('top_n', 1)
    api_key = data.get('api_key', os.environ.get('OPENAI_API_KEY'))

    if database_name not in database_configs:
        valid_databases = ", ".join(database_configs.keys())
        return jsonify({
            "error": f"Invalid database name: {database_name}. Valid database names are: {valid_databases}"
        }), 400

    openai.api_key = api_key
    config = database_configs[database_name]
    products_df = data_frames[config["df"]]
    query_embedding = get_query_embedding(product_description)
    top_matches = find_top_matches(products_df, query_embedding, top_n=top_n)
    response_data = [
        format_response(row[1], config, row[1]["similarity"], amount)
        for row in top_matches.iterrows()
    ]
    return jsonify(response_data)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

def choose_top_n_matches(matches):
    print(1)