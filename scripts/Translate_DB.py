from transformers import MarianMTModel, MarianTokenizer
import torch
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Initialize tokenizer and model globally to avoid reloading in each subprocess
model_name = "Helsinki-NLP/opus-mt-fr-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
if torch.cuda.is_available():
    model = model.to('cuda')

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

def process_chunk(chunk,columns_to_translate):
    for column in columns_to_translate:
        # Process each column in batches
        chunk_size = 10  # Define your batch size; adjust based on your memory and performance needs
        translated_texts = []
        for i in range(0, len(chunk), chunk_size):
            batch_texts = chunk[column].iloc[i:i + chunk_size].tolist()
            translated_texts.extend(translate_batch(batch_texts))
        chunk[f'{column}_translated'] = translated_texts
    return chunk

def main(df, columns_to_translate):
    # Split DataFrame into chunks
    num_splits = 10  # Adjust based on your system's memory and CPU cores
    chunks = np.array_split(df, num_splits)

    # Use ProcessPoolExecutor to parallelize the translation
    with ProcessPoolExecutor(max_workers=num_splits) as executor:
        futures = [executor.submit(process_chunk, chunk,columns_to_translate) for chunk in chunks]
        processed_chunks = [f.result() for f in futures]

    # Concatenate the processed chunks back into a single DataFrame
    translated_df = pd.concat(processed_chunks)
    return translated_df

if __name__ == '__main__':
    # Example DataFrame loading
    df = pd.read_pickle(r"data\NACRES_with_embeddings_and_factors.pkl") # Load the DataFrame  
    columns_to_translate = ["Segment"]  # List your columns here
    df = df.head(10)
    translated_df = main(df, columns_to_translate)
    # Save or use your translated DataFrame
    #translated_df.to_csv(r"data\NACRES_with_embeddings_and_factors_translated.csv", index=False)



