from transformers import MarianMTModel, MarianTokenizer
import torch
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import numpy as np

# Initialize tokenizer and model globally
model_name = "Helsinki-NLP/opus-mt-fr-en"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

if torch.cuda.is_available():
    model = model.to('cuda')

def translate_batch(texts):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    encoded_inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        translated_tokens = model.generate(**encoded_inputs)
    translated_texts = [tokenizer.decode(t, skip_special_tokens=True) for t in translated_tokens]
    return translated_texts

def process_chunk(chunk, columns_to_translate, suffix):
    for column in columns_to_translate:
        chunk_size = 10
        translated_texts = []
        for i in range(0, len(chunk), chunk_size):
            batch_texts = chunk[column].iloc[i:i + chunk_size].tolist()
            translated_texts.extend(translate_batch(batch_texts))
        chunk[column + suffix] = translated_texts
    return chunk

def translate(df, columns_to_translate, suffix="_translated"):
    num_splits = 10
    chunks = np.array_split(df, num_splits)
    with ProcessPoolExecutor(max_workers=num_splits) as executor:
        futures = [executor.submit(process_chunk, chunk, columns_to_translate, suffix) for chunk in chunks]
        processed_chunks = [f.result() for f in futures]
    translated_df = pd.concat(processed_chunks)
    return translated_df

def translate_DB(input_data, columns_to_translate=None, output_file_name=None):
    if isinstance(input_data, str):
        # Handle string input
        translated_text = translate_batch([input_data])[0]
        result = {columns_to_translate[0]: input_data, f"{columns_to_translate[0]}_translated": translated_text}
        return pd.DataFrame([result])

    elif isinstance(input_data, pd.DataFrame):
        df = input_data
    else:
        raise ValueError("Input must be a string or a DataFrame")

    if not columns_to_translate:
        Warning("No columns to translate were provided, no translation will be performed.")
        return df

    translated_df = translate(df, columns_to_translate, suffix="_translated")
    for column in columns_to_translate:
        translated_df[f"{column} original"] = translated_df[column]
        translated_df[column] = translated_df[f"{column}_translated"]
        translated_df.drop(columns=[f"{column}_translated"], inplace=True)

    if output_file_name:
        translated_df.to_pickle(output_file_name, index=False)

    return translated_df

if __name__ == '__main__':
    path = r"data\achats_EPFL\Test_100_articles.xlsx"
    columns_to_translate = ["Désignation article", "Famille"]
    df = pd.read_excel(path)
    translated_df = translate_DB(df, columns_to_translate)
    translated_df.to_excel(r"data\achats/translated_test_100_articles.xlsx", index=False)
