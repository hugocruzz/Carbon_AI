import requests
import pandas as pd
from datetime import datetime

# Function to fetch exchange rates for a specific date
def get_exchange_rate(date, currency, target_currency='EUR'):
    url = f'https://api.frankfurter.app/{date}?from={currency}&to={target_currency}'
    response = requests.get(url)
    data = response.json()
    if currency==target_currency:
        return 1
    if 'rates' not in data:
        if currency=="XOF":
            return 0.00150
        raise ValueError('Error fetching exchange rate')
    return data['rates'][target_currency]

# Function to fetch exchange rates for a specific month and currency
def get_monthly_exchange_rate(year_month, currency, target_currency='EUR'):
    date = f'{year_month}-01'
    return get_exchange_rate(date, currency, target_currency=target_currency)

def currency_converter(df, target_currency, date_column, currency_column, amount_column):
    # Convert 'Date de commande' to datetime
    df[date_column] = pd.to_datetime(df[date_column], format='%d/%m/%Y')

    # Create a 'Year-Month' column
    df['Year-Month'] = df[date_column].dt.to_period('M')

    # Get unique Year-Month and Devise combinations
    unique_combinations = df[['Year-Month', currency_column]].drop_duplicates()

    # Fetch exchange rates for each unique Year-Month and Devise combination
    exchange_rates = {}
    for _, row in unique_combinations.iterrows():
        year_month = row['Year-Month'].strftime('%Y-%m')
        currency = row[currency_column]
        exchange_rate = get_monthly_exchange_rate(year_month, currency, target_currency=target_currency)
        exchange_rates[(year_month, currency)] = exchange_rate

    # Conversion function
    def convert_to_euros_optimized(row):
        year_month = row['Year-Month'].strftime('%Y-%m')
        currency = row[currency_column]
        price = row[amount_column]

        # Get the exchange rate for the given Year-Month and currency
        rate = exchange_rates[(year_month, currency)]

        # Convert to euros
        price_in_euros = price * rate
        return price_in_euros

    # Apply conversion
    df[f'Amount in {target_currency}'] = df.apply(convert_to_euros_optimized, axis=1)

    return df
