# Carbon AI 

Do you need to estimate the CO2 emission of a large dataset containing low quality description of articles ? This project is for you.
We use AI (NLP) to translate and embed the description of the articles and then we match them with a carbon database to estimate the CO2 emission of each article.
For now, this project is focused on academic data and therefore uses the Labo1point5 database for emission factors.

## Project Overview

This project takes a dataframe of articles and calculate the CO2 impact using the Labo1point5 database for emission factors.

## Data science overview
The project aims to automate the process of integrating and matching datasets using natural language processing (NLP) techniques. It involves translating textual descriptions, embedding them into vector representations, and matching them against a target dataset to find closest matches. This pipeline is useful for applications such as product matching across different databases or systems.

## Features

- **Translation**: Utilizes machine translation models to translate textual data from one language to another.
- **Embedding**: Converts textual data into numerical embeddings using pre-trained models.
- **Matching**: Matches datasets based on cosine similarity between embeddings.
- **Output**: Generates matched datasets and calculates additional metrics like CO2 emissions based on user inputs.

## Requirements

Ensure you have the following installed:

- Python 3.x
- Required Python libraries (`requirements.txt`)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/hugocruzz/Carbon_AI.git
   cd Carbon_AI
   ```

2. Install dependencies:

```pip install -r requirements.txt```


3. Obtain necessary API keys and place them in appropriate locations to open them in the scripts:
- FRED API key: https://fred.stlouisfed.org/
- OpenAI API key: https://beta.openai.com/account/api-keys

4. Modify the config file to match your desires

### Step 1: Translate and Embed Data

Modify `main_string.py` or `main.py` to accept either a string input or a DataFrame containing articles. Adjust columns and paths (input_path.yaml) as necessary.

### Step 2: Match Datasets

Run `main_string.py` to:
- Translate and embed the source data.
- Load and process the target dataset.
- Match datasets based on embeddings using `Match_datasets.py`.

### Step 3: Adjust and Customize

Refine translations, embeddings, or matching strategies based on specific requirements. Adjust parameters like batch sizes and model configurations in the scripts as needed.

### Example

Here's how to run the main pipeline for a string input:

```python main.py```


Ensure to adjust configurations and paths in the scripts (`main.py`, `Translate_DB.py`, `Embed_dataframe.py`, `Match_datasets.py`) based on your environment and dataset characteristics.

## License

This project is licensed under the MIT License - see the [LICENSE] file for details.

## Acknowledgments

- Inspired by the need for efficient data integration and matching techniques in real-world applications.

## Troubleshooting

- If encountering issues with dependencies or scripts, refer to the troubleshooting section in the README or contact the repository owner for assistance.
