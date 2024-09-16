import pandas as pd 
import numpy as np
import os
from openai import OpenAI
import logging 
import json

def chat_bot_structured_input(api_key=None, model = "gpt-4o-mini"):
    example_json = {"Description": "Laptop Dell XPS",
                        "additional_context": "Computers",
                        "unit": "kg",
                        "value": 2.5,
                        "Allocation_method": "Physical allocation",  #// Assuming this is typical for such products, but can vary.
                        "Market_transformation": "Market",#//// Assuming the emission factor is based on market-level data.
                        "Geographical_scope": "Global",  #//// Laptops are generally assessed globally unless specified otherwise.
                        "Time_period": "2024",  #//// Update to the most recent or relevant year for emission data.
                        "System_boundary": "Cradle to grave", #// // Covers the full lifecycle from raw material extraction to disposal.
                        "Unit_measurement": "kg CO2 eq per laptop",  #//// Adapting the unit based on the provided value.
                        "Cutoff": "Partial cutoff", #// // Electronics often use partial cutoff depending on upstream processes considered.
                        "Environmental_impact": "Global warming potential"  #//// Common impact category for electronic products.
                        }
    
    input_descriptions_json = {
    "Description": "Concise description of the product",
    "additional_context": "Additional context to help understand the product, if not provided, leave it empty",
    "unit": "Unit of the value (standard units: kg, m3, MJ, kWh, m, m2, hours, year, g). Convert to the standard unit if necessary and adapt the value accordingly.",
    "value": "Value of the product in the unit provided",

    "Allocation_method": "Allocation method used for the LCA. Options include 'Economic allocation', 'Physical allocation', 'No allocation', 'Substitution'. You can set the default based on the unit of the value you found or if unknown: 'No allocation'.",
    "Market_transformation": "Market transformation considered in the LCA. Options include 'Market' (market-based approach) and 'Transformation' (focus on transformation processes). Default if unknown: 'Market'.",
    "Geographical_scope": "Geographical scope of the LCA. Options include specific regions such as 'Europe', 'United States', 'Asia', or 'Global'. Default if unknown: 'Global'.",
    "Time_period": "Time period of the LCA. Indicate the year or range of years for which the data is relevant. Default if unknown: '2024'.",
    "System_boundary": "System boundary of the LCA. Options include 'Cradle to gate' (from raw material extraction to product output), 'Cradle to grave' (full lifecycle including disposal), 'Gate to gate' (within a single facility), or 'Well to wheel' (for transportation). Default if unknown: 'Cradle to grave'.",
    "Unit_measurement": "Unit of measurement of the LCA. Common units include 'kg CO2 eq per kg', 'g CO2 eq per kWh', etc. Adapt based on the specific emission factor used.",
    "Cutoff": "Cutoff method used in the LCA. Options include 'Cumulative cutoff' (includes all upstream processes) and 'Cutoff' (excludes some upstream processes). Default if unknown: 'Cumulative cutoff'.",
    "Environmental_impact": "Environmental impact considered in the LCA. Common impacts include 'Global warming potential', 'Acidification potential', 'Eutrophication potential', etc. Default if unknown: 'Global warming potential'."
    }
    
    prompt = f"""Your an expert in Life Cycle analysis and you're very comfortable vulgarizing concepts.
    Your job is to format the information the user is giving you to fill a dictionnary containing the following keys: {json.dumps(input_descriptions_json)}.

    Some informations can be filled using the user inputs, some others can be set as default values.
    If some informations is important, feel free to ask the user for more details.

    Provide your response in a valid JSON format following this schema: {json.dumps(example_json)}
    """

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"]) or OpenAI(api_key=api_key)
    success = False

    while not success:
        chat_completion = client.chat.completions.create(
                    model=model,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": f"""Can you give me a description following the 
                         LCA characteristics of the following product: 3 g of pencils, Office supply, 
                         i want the ecotoxicity and the whole process including the transportation to France. 
                         I don't want to take into account recycling"""}
                    ]
                )
        
        data = chat_completion.choices[0].message.content
        data_json = json.loads(data)
        success = True
    return data_json

if __name__ == '__main__':
    chat_bot_structured_input()