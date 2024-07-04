from transformers import MarianMTModel, MarianTokenizer
import torch
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def translate_batch(texts):
    # Ensure the function uses the globally loaded model and tokenizer
    global tokenizer, model

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Prepare the texts for translation and move to the appropriate device
    encoded_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():  # Ensure no gradients are calculated
        translated_tokens = model.generate(**encoded_inputs)
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
    return translated_texts

def process_chunk(chunk,columns_to_translate, suffix):
    for column in columns_to_translate:
        # Process each column in batches
        chunk_size = 10  # Define your batch size; adjust based on your memory and performance needs
        translated_texts = []
        for i in range(0, len(chunk), chunk_size):
            batch_texts = chunk[column].iloc[i:i + chunk_size].tolist()
            translated_texts.extend(translate_batch(batch_texts))
        chunk[column+suffix] = translated_texts
    return chunk

def translate(df, columns_to_translate, suffix="_translated"):
    # Split DataFrame into chunks
    num_splits = 10  # Adjust based on your system's memory and CPU cores
    chunks = np.array_split(df, num_splits)

    # Use ProcessPoolExecutor to parallelize the translation
    with ProcessPoolExecutor(max_workers=num_splits) as executor:
        futures = [executor.submit(process_chunk, chunk,columns_to_translate, suffix) for chunk in chunks]
        processed_chunks = [f.result() for f in futures]

    # Concatenate the processed chunks back into a single DataFrame
    translated_df = pd.concat(processed_chunks)
    return translated_df

def translate_DB(df, columns_to_translate, output_file_name=None):
    #Check if columns_to_translate is empty
    if not columns_to_translate:
        Warning("No columns to translate were provided, no translation will be performed.")
        return df
    # Initialize tokenizer and model globally to avoid reloading in each subprocess
    model_name = "Helsinki-NLP/opus-mt-fr-en"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    
    if torch.cuda.is_available():
        model = model.to('cuda')
    # Update the DataFrame with original and translated columns
    translated_df = translate(df, columns_to_translate, suffix="_translated")
    for column in columns_to_translate:
        translated_df[f"{column} original"] = translated_df[column]
        translated_df[column] = translated_df[f"{column}_translated"]
        translated_df.drop(columns=[f"{column}_translated"], inplace=True)
    if output_file_name:
        translated_df.to_pickle(output_file_name, index=False)
    return translated_df

if __name__ == '__main__':
    # Example DataFrame loading
    path = r"data\achats_EPFL\Test_100_articles.xlsx"
    columns_to_translate = ["Désignation article", "Famille"]  # List your columns here
    df = pd.read_excel(path) # Load the DataFrame  
    translated_df = translate_DB(path, columns_to_translate)
    translated_df.to_excel(r"data\achats/translated_test_100_articles.xlsx", index=False)
    