# Carbon AI

# API test/Usage:

### Test the API without python

You can test the API by using an online API testing tool like https://reqbin.com/ :
URL: https://carbon-ai.onrender.com/search
Method: POST
Format: Json
Dictionnary:
{
    "product_description": "Ordinateur de bureau",  
    "api_key": "insert your API key here",
    "amount": 1000
}

### Test the API with Python
1. Python:
```python
import requests

# The URL where your Flask API is running
api_url = "https://carbon-ai.onrender.com/search"

# The data you want to send to the API (example values)
data = {
    "product_description": "Ordinateur de bureau",  # Example product description
    "api_key": "insert your API key here",
    "amount": 1000,  # Example price,
}

# Make a POST request to the API
response = requests.post(api_url, json=data)

# Check if the request was successful
if response.status_code == 200:
    # Get the JSON response data
    result = response.json()
    print("API Response:", result)
else:
    print("Failed to call API:", response.status_code, response.text)

```
Results:
```python
API Response: {'CO2_emitted': 430.0, 'CO2_unit': 'kg CO2e', 'emission_factor': 0.43, 'emission_factor_unit': 'kg CO2e per euro spent', 'input': 'Ordinateur de bureau', 'matched_NACRES_code': 'IA01', 'matched_category': 'MICRO-ORDINATEURS ET STATIONS DE TRAVAIL FIXES', 'similarity_score': 0.85}
```

## Instructions for developper
1. Clone the repository
2. Run the following command to install the required packages:
```bash
git clone git@github.com:hugocruzz/Carbon_AI.git
pip install -r requirements.txt
```

## Usage:
1. scripts/CO2_calculator_dataframe.py takes import a csv file, creates an ebedded vector of the columns specified as input. Then use the Labo1.5 database to compute the CO2 emissions of the articles in the csv file. The output is a csv file with the CO2 emissions of the articles.
2. scripts/CO2_calculator_string_API.py is the source script of the API. 

## Versions:
Version: 0.1.0:
- Initial release
- Develop the API taking a string as input, interpret and returning the CO2 emissions of the product
- Takes into account multiple database for the CO2 emissions, like Agrybalise, mobitool etc
- Use Top_n as arguments 

Future releases:
- Chatbot that will ask the user for the product description and price
- Ajouter couche AI pour interpreter l'input.
### API parameters:
Inputs:
- product_description: string in which the product description is written
- price: float in which the price of the product is written
- api_key: string in which the API key is written

Outputs:
- CO2_emitted: float, the CO2 emissions of the product if a price is indicated
- CO2_unit: string, the unit of the CO2 emissions
- emission_factor: float, the emission factor of the product
- emission_factor_unit: string, the unit of the emission factor
- input: string, the input product description
- matched_NACRES_code: string, the NACRES code of the product
- matched_category: string, the category of the product (Intitulé NACRES)
- similarity_score: float, the similarity score of the product with the database, the highest the better match
```

Tester avec C1333 labo
Check avec 2023