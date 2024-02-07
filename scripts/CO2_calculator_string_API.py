from flask import Flask, request, jsonify
import os
import pandas as pd
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

app = Flask(__name__)

limiter = Limiter(
    app,
    key_func=get_remote_address,  # Use the remote address of the client as the key for rate limiting
    default_limits=["5 per day"]  # Set the rate limit to 5 requests per day
)

# Function to search in DataFrame
def search_in_df(df, product_description, n=3):
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002",
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))
    results_score = df["similarity"].max()
    results_index = df["similarity"].argmax()
    return results_index, results_score

@limiter.limit("5 per day")  # Apply the rate limit to this endpoint
@app.route('/search', methods=['POST'])
@app.errorhandler(429)
def ratelimit_handler(e):
    return "You have exceeded your daily request limit. Please try again tomorrow.", 429

def search():
    data = request.json
    input_string = data.get('input_string')
    input_price = float(data.get('input_price'))
    API_key = data.get('API_key')
    if API_key is None:
        API_key = os.environ.get('OPENAI_API_KEY')
        
    # Set API key from environment variable
    openai.api_key = API_key

    # Load DataFrame (adjust path as needed)
    path_database = "data/processed/nacres/NACRES_with_embeddings_and_factors.pkl"
    df = pd.read_pickle(path_database)

    # Search in DataFrame
    search_output = search_in_df(df, input_string, n=1)
    df_object = df.iloc[search_output[0]]
    
    # Perform the calculation
    calculation_result = df_object["FEL1P5"] * input_price
    similarity_score = int(search_output[1]*100)/100

    # Prepare the result
    result = {
        'description': f"'{input_string}' is categorized as '{df_object['Intitulés Nacres']}' (Classe: {df_object['Classe']}).",
        'similarity_score': similarity_score,
        'calculation_result': calculation_result
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
