import pandas as pd
import math
import numpy as np

MODEL_NAME = "gpt-4.1"
df = pd.read_csv(f'./final_result/{MODEL_NAME}_equal_evidence.csv')

# sector 없이 ticker+name별 집계
ticker_grouped = df.groupby(['ticker', 'name']).agg(
    buy_count=('llm_answer', lambda x: (x.str.lower() == 'buy').sum()),
    sell_count=('llm_answer', lambda x: (x.str.lower() == 'sell').sum())
).reset_index()

ticker_grouped['total_count'] = ticker_grouped['buy_count'] + ticker_grouped['sell_count']
ticker_grouped['preference'] = (ticker_grouped['sell_count'] - ticker_grouped['buy_count']).abs() / ticker_grouped['total_count']
ticker_grouped['buy_ratio'] = ticker_grouped['buy_count'] / ticker_grouped['total_count']
ticker_grouped['biased'] = np.where(ticker_grouped['buy_ratio'] >= 0.7, 'buy', 'sell')

total_stocks = len(ticker_grouped)
percentiles = [0.05]  # 1%, 5%, 10%

for p in percentiles:
    n = max(1, math.ceil(total_stocks * p))
    # 상위 p%
    top_pref_df = ticker_grouped.sort_values('preference', ascending=False).head(n)
    top_pref_df = top_pref_df[['ticker', 'name', 'buy_count', 'sell_count', 'total_count', 'buy_ratio', 'preference', 'biased']]
    top_pref_df = top_pref_df.reset_index(drop=True)
    top_fname = f"./final_result/{MODEL_NAME}_top_{int(p*100)}_preference.csv"
    top_pref_df.to_csv(top_fname, index=False)
    print(f"상위 {int(p*100)}% ({n}개) 저장: {top_fname}")
    print(top_pref_df.head(5))
    
    # # 하위 p%
    # bottom_pref_df = ticker_grouped.sort_values('preference', ascending=True).head(n)
    # bottom_pref_df = bottom_pref_df[['ticker', 'name', 'buy_count', 'sell_count', 'total_count', 'buy_ratio', 'preference']]
    # bottom_pref_df = bottom_pref_df.reset_index(drop=True)
    # bottom_fname = f"./{MODEL_NAME}_bottom_{int(p*100)}_preference.csv"
    # bottom_pref_df.to_csv(bottom_fname, index=False)
    # print(f"하위 {int(p*100)}% ({n}개) 저장: {bottom_fname}")
    # print(bottom_pref_df.head(5))