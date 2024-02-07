from flask import Flask, request, render_template_string, session
import openai
import os
import pandas as pd
import numpy as np

from openai.embeddings_utils import get_embedding, cosine_similarity
from flask_session import Session  # Install Flask-Session if not already installed

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Set a secret key for session management
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

results = []  # Store previous results

def search_in_df(df, product_description, n=3, pprint=True):
    product_embedding = get_embedding(
        product_description,
        engine="text-embedding-ada-002",
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, product_embedding))
    results_score = df["similarity"].max()
    results_index = df["similarity"].argmax()
    return (results_index, results_score)

@app.route('/', methods=['GET', 'POST'])
def main():
    if request.method == 'POST':
        input_string = request.form['input_string']
        input_price = float(request.form['input_price'])  # Get the input price as float
        api_key = request.form.get('api_key')
        if api_key:  # If an API key is provided, store it in the session
            session['api_key'] = api_key
        elif 'api_key' in session:  # Use the API key stored in the session
            api_key = session['api_key']

        openai.api_key = api_key
        os.environ["OPENAI_API_KEY"] = api_key

        path_database = r"data\processed\nacres\NACRES_with_embeddings_and_factors.pkl"

        df = pd.read_pickle(path_database)
        search_output = search_in_df(df, input_string, n=1, pprint=False)
        df_object = df.iloc[search_output[0]]

        # Perform the calculation using the "FEL1P5" field from the df_object and the input price
        calculation_result = df_object["FEL1P5"] * input_price
        similarity_score = int(search_output[1]*100)/100
        result = (f"'{input_string}' is categorized as '{df_object['Intitulés Nacres']}' "
                  f"(Classe: {df_object['Classe']}). \nThe similarity score is {similarity_score}. \n"
                  f"Calculation Result: {calculation_result} kg CO2e.")

        results.append(result)

    return render_template_string('''
        <html>
            <head>
                <style>
                    body { 
                        text-align: center; 
                        margin-top: 50px; 
                    }
                    form { 
                        display: inline-block; 
                        margin-bottom: 20px; 
                    }
                    input[type="text"], input[type="password"], input[type="number"] { 
                        margin-bottom: 10px; 
                        width: 300px; 
                    }
                    input[type="submit"] { 
                        width: 150px; 
                        height: 40px; 
                        font-size: 16px; 
                    }
                    .results { 
                        text-align: left; 
                        display: inline-block; 
                        text-align: left;
                    }
                </style>
            </head>
            <body>
                <form method="post">
                    API Key: <input type="password" name="api_key" placeholder="Enter API Key" {{'value="' + session['api_key'] + '"' if 'api_key' in session else ''}}><br>
                    Input String: <input type="text" name="input_string"><br>
                    Prix (euros): <input type="number" name="input_price" step="0.01" placeholder="0.00"><br>
                    <input type="submit" value="Submit">
                </form>
                <div class="results">
                    {% for result in results %}
                        <p>{{ result }}</p>
                    {% endfor %}
                </div>
            </body>
        </html>
    ''', results=results)

if __name__ == '__main__':
    app.run(debug=True)
