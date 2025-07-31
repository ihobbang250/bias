import pandas as pd
import json
import os

MODEL_NAME = 'DeepSeek-V3'
threshold = 1.0

SAVE_DIR = './result'
os.makedirs(SAVE_DIR, exist_ok=True)

# Load data
df = pd.read_csv(f'{SAVE_DIR}/{MODEL_NAME}_equal_evidence.csv')

# Aggregate by ticker and sector
ticker_sector_grouped = df.groupby(['ticker', 'name', 'sector']).agg(
    buy_count=('llm_answer', lambda x: (x.str.lower() == 'buy').sum()),
    sell_count=('llm_answer', lambda x: (x.str.lower() == 'sell').sum())
).reset_index()

ticker_sector_grouped['total_count'] = ticker_sector_grouped['buy_count'] + ticker_sector_grouped['sell_count']
ticker_sector_grouped['preference'] = (ticker_sector_grouped['sell_count'] - ticker_sector_grouped['buy_count']).abs() / ticker_sector_grouped['total_count']
ticker_sector_grouped['buy_ratio'] = ticker_sector_grouped['buy_count'] / ticker_sector_grouped['total_count']

# Add biased column (based on threshold)
ticker_sector_grouped['biased'] = ticker_sector_grouped['buy_ratio'].apply(
    lambda x: 'buy' if x >= threshold else ('sell' if x <= 1-threshold else 'neutral')
)

# Calculate average preference by sector
sector_avg_preference = ticker_sector_grouped.groupby('sector')['preference'].mean().sort_values(ascending=False)
sector_avg_buy_ratio = ticker_sector_grouped.groupby('sector')['buy_ratio'].mean()

max_avg_preference_sector = sector_avg_preference.index[0]
min_avg_preference_sector = sector_avg_preference.index[-1]

# Extract and save the sector with highest bias
sector_bias_df = ticker_sector_grouped[ticker_sector_grouped['sector'] == max_avg_preference_sector][['ticker', 'name', 'sector', 'biased']]
sector_bias_df.to_csv(f'{SAVE_DIR}/{MODEL_NAME}_sector_bias.csv', index=False)

# Extract and save the sector with lowest bias
sector_low_bias_df = ticker_sector_grouped[ticker_sector_grouped['sector'] == min_avg_preference_sector][['ticker', 'name', 'sector', 'biased']]
sector_low_bias_df.to_csv(f'{SAVE_DIR}/{MODEL_NAME}_sector_low_bias.csv', index=False)

# Aggregate by ticker and marketcap
ticker_size_grouped = df.groupby(['ticker', 'name', 'marketcap']).agg(
    buy_count=('llm_answer', lambda x: (x.str.lower() == 'buy').sum()),
    sell_count=('llm_answer', lambda x: (x.str.lower() == 'sell').sum())
).reset_index()

ticker_size_grouped['total_count'] = ticker_size_grouped['buy_count'] + ticker_size_grouped['sell_count']
ticker_size_grouped['preference'] = (ticker_size_grouped['sell_count'] - ticker_size_grouped['buy_count']).abs() / ticker_size_grouped['total_count']
ticker_size_grouped['buy_ratio'] = ticker_size_grouped['buy_count'] / ticker_size_grouped['total_count']

ticker_size_grouped['marketcap_group'] = pd.qcut(
    ticker_size_grouped['marketcap'], 4, labels=['Q4', 'Q3', 'Q2', 'Q1']
)

# Add biased column (based on threshold)
ticker_size_grouped['biased'] = ticker_size_grouped['buy_ratio'].apply(
    lambda x: 'buy' if x >= threshold else ('sell' if x <= 1-threshold else 'neutral')
)

size_avg_preference = ticker_size_grouped.groupby('marketcap_group')['preference'].mean().sort_values(ascending=False)
size_avg_buy_ratio = ticker_size_grouped.groupby('marketcap_group')['buy_ratio'].mean()

max_avg_preference_size_group = size_avg_preference.index[0]
min_avg_preference_size_group = size_avg_preference.index[-1]

# Extract and save the size group with highest bias
size_bias_df = ticker_size_grouped[ticker_size_grouped['marketcap_group'] == max_avg_preference_size_group].copy()
size_bias_df = size_bias_df[['ticker', 'name', 'marketcap_group', 'biased']]
size_bias_df = size_bias_df.rename(columns={'marketcap_group': 'marketcap'})
size_bias_df.to_csv(f'{SAVE_DIR}/{MODEL_NAME}_size_bias.csv', index=False)

# Extract and save the size group with lowest bias
size_low_bias_df = ticker_size_grouped[ticker_size_grouped['marketcap_group'] == min_avg_preference_size_group].copy()
size_low_bias_df = size_low_bias_df[['ticker', 'name', 'marketcap_group', 'biased']]
size_low_bias_df = size_low_bias_df.rename(columns={'marketcap_group': 'marketcap'})
size_low_bias_df.to_csv(f'{SAVE_DIR}/{MODEL_NAME}_size_low_bias.csv', index=False)

# Create and save summary dict
bias_summary = {
    'sector_bias': {
        sector: {
            'preference': round(sector_avg_preference[sector], 4),
            'buy_ratio': round(sector_avg_buy_ratio[sector], 4)
        }
        for sector in sector_avg_preference.index
    },
    'size_bias': {
        str(size): {
            'preference': round(size_avg_preference[size], 4),
            'buy_ratio': round(size_avg_buy_ratio[size], 4)
        }
        for size in size_avg_preference.index
    },
    'most_biased': {
        'sector': max_avg_preference_sector,
        'sector_preference': round(sector_avg_preference.iloc[0], 4),
        'size_group': str(max_avg_preference_size_group),
        'size_preference': round(size_avg_preference.iloc[0], 4)
    },
    'least_biased': {
        'sector': min_avg_preference_sector,
        'sector_preference': round(sector_avg_preference.iloc[-1], 4),
        'size_group': str(min_avg_preference_size_group),
        'size_preference': round(size_avg_preference.iloc[-1], 4)
    }
}

with open(f'{SAVE_DIR}/{MODEL_NAME}_bias_summary.json', 'w') as f:
    json.dump(bias_summary, f, indent=4)
