import requests
import pandas as pd
from datetime import datetime
from fredapi import Fred
import os

# Function to fetch HICP data for the Eurozone
def get_hicp_data(fred, start_date, end_date):
    hicp = fred.get_series('CP0000EZ19M086NEST', observation_start=start_date, observation_end=end_date)
    return hicp

# Function to calculate the inflation factor
def calculate_inflation_factor(fred, current_date, historical_date):
    # Fetch HICP data for the range including both dates
    start_date = min(current_date, historical_date)
    end_date = max(current_date, historical_date)
    
    hicp_data = get_hicp_data(fred, start_date, end_date)
    
    # Get the HICP values for the specific dates
    hicp_current = hicp_data[current_date]
    hicp_historical = hicp_data[historical_date]
    
    # Calculate the inflation factor
    inflation_factor = hicp_current / hicp_historical
    return inflation_factor

# Function to fetch exchange rates for a specific date
def get_exchange_rate(date, currency, target_currency='EUR'):
    url = f'https://api.frankfurter.app/{date}?from={currency}&to={target_currency}'
    response = requests.get(url)
    data = response.json()
    if currency == target_currency:
        return 1
    if 'rates' not in data:
        if currency == "XOF":
            return 0.00150
        raise ValueError('Error fetching exchange rate')
    return data['rates'][target_currency]

# Function to fetch exchange rates for a specific month and currency
def get_monthly_exchange_rate(year_month, currency, target_currency='EUR'):
    date = f'{year_month}-01'
    return get_exchange_rate(date, currency, target_currency=target_currency)

def currency_converter(df, target_currency, date_column, currency_column, amount_column, target_inflation_year, fred_api_key=os.getenv('FRED')):
    # Replace 'YOUR_API_KEY' with your actual FRED API key
    fred = Fred(api_key=fred_api_key)
    #Convert target_inflation_year to string if it is not
    target_inflation_year = str(target_inflation_year)
    # Convert 'Date de commande' to datetime
    df[date_column] = pd.to_datetime(df[date_column], format='%d/%m/%Y')

    # Create a 'Year-Month' column
    df['Year-Month'] = df[date_column].dt.to_period('M')

    # Get unique Year-Month and Devise combinations
    unique_combinations = df[['Year-Month', currency_column]].drop_duplicates()

    # Fetch exchange rates and inflation factors for each unique Year-Month and Devise combination
    exchange_rates = {}
    inflation_factors = {}
    for _, row in unique_combinations.iterrows():
        year_month = row['Year-Month'].strftime('%Y-%m')
        currency = row[currency_column]
        exchange_rate = get_monthly_exchange_rate(year_month, currency, target_currency=target_currency)
        exchange_rates[(year_month, currency)] = exchange_rate

        # Calculate inflation factor for the given year_month
        inflation_factor = calculate_inflation_factor(fred, f'{year_month}-01', f'{target_inflation_year}-01-01')
        inflation_factors[year_month] = inflation_factor

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

    # Function to adjust price for inflation
    def adjust_for_inflation(row):
        year_month = row['Year-Month'].strftime('%Y-%m')
        price_in_euros = row[f'Amount in {target_currency}']
        inflation_factor = inflation_factors[year_month]
        adjusted_price = price_in_euros / inflation_factor
        return adjusted_price

    # Apply conversion
    df[f'Amount in {target_currency}'] = df.apply(convert_to_euros_optimized, axis=1)
    df[f'Amount in {target_currency} corrected inflation in {target_inflation_year}'] = df.apply(adjust_for_inflation, axis=1)
    
    return df[f'Amount in {target_currency}'], df[f'Amount in {target_currency} corrected inflation in {target_inflation_year}']
