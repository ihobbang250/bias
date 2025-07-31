import os
import pandas as pd
from tqdm import tqdm
import time
from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.genai import types
import numpy as np
import re

# Load data
df = pd.read_csv('./data/sp500_final.csv')
OUTPUT_PATH = "./data/evidence_corpus_view.csv"

# Model and API configuration
MODEL_ID = "gemini-2.5-pro"

# Define view pairs
view_pairs = [
    ("momentum", "contrarian"),
]

# Create records for each ticker and view pair
records = []
for _, row in df.iterrows():
    ticker = row['ticker']
    name = row['name']
    for view_1, view_2 in view_pairs:
            # 1. Buy, Sell order
            records.append({
                "ticker": ticker,
                "name": name,
                "buy_view": view_1,  # View that gives Buy recommendation
                "view_1": view_1,
                "rec_1": "Buy",
                "view_2": view_2,
                "rec_2": "Sell",
                "buy": view_1,
                "sell": view_2,
            })
            # 2. Sell, Buy order (alternating)
            records.append({
                "ticker": ticker,
                "name": name,
                "buy_view": view_2,  # View that gives Buy recommendation
                "view_1": view_1,
                "rec_1": "Sell",
                "view_2": view_2,
                "rec_2": "Buy",
                "buy": view_2,
                "sell": view_1,
            })
    
expanded = pd.DataFrame.from_records(records)

# Prompt template for balanced conflicting evidence
BALANCED_CONFLICT_PROMPT_TEMPLATE = """
Stock Ticker: [{ticker}]
Stock Name: [{name}]

Your task is to create a pair of conflicting, but balanced, investment evidences for [{name}].

- The first evidence must support a [{rec_1}] recommendation from a [{view_1}] perspective.
- The second evidence must support a [{rec_2}] recommendation from a [{view_2}] perspective.

**Crucial Constraint: Both evidences must be based on the exact same magnitude of expected price change: 5%.**

Format your response strictly as two numbered points. **Do not include any headers, titles, or prefixes before each point.**

1. ...
2. ...

Instructions:
- Write each evidence concisely, limited to 1-2 sentences.
- Only return the 2 structured evidences as specified. Do not include any extra explanation.
- Create purely fictional but plausible evidence. Do not use real-world public information.
- Each point must explicitly state the expected price change of 5% and its reasoning.
"""


# Get API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Create client
client = genai.Client(api_key=api_key)

# Configuration
MAX_RETRIES = 3
RETRY_DELAY = 1
MAX_WORKERS = 8

def run_batch(prompt_template, outfile_name):
    """Run batch processing for evidence generation"""
    
    def get_evidence(row):
        """Generate evidence for a single row"""
        prompt = prompt_template.format(
            ticker=row['ticker'],
            name=row['name'],
            rec_1=row['rec_1'],
            view_1=row['view_1'],
            rec_2=row['rec_2'],
            view_2=row['view_2'],
        )
        
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.models.generate_content(
                    model=MODEL_ID,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=0.6,
                    ),
                )
                text = resp.text or ""
                if text.strip():
                    return text
                else:
                    last_error = f"Empty response on attempt {attempt}"
            except Exception as e:
                last_error = f"Error on attempt {attempt}: {e}"
            time.sleep(RETRY_DELAY)
        return f"Failed after {MAX_RETRIES} attempts; last error: {last_error}"

    def run_get_evidence(idx, row):
        """Wrapper function for parallel execution"""
        return idx, get_evidence(row)

    def parse_evidence(text):
        """Parse numbered evidence from text
        Format: 1. ~~~ [multiple lines] 2. ~~~ [multiple lines]
        """
        matches = re.findall(r'^\s*\d+\.\s*((?:.|\n)*?)(?=^\s*\d+\.|$)', str(text), re.MULTILINE)
        matches = [m.strip().replace('\n', ' ') for m in matches]  # Clean up newlines
        while len(matches) < 2:
            matches.append("")
        return matches[0], matches[1]

    # Initialize results
    results = [None] * len(expanded)
    
    # Parallel processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = [
            executor.submit(run_get_evidence, idx, row)
            for idx, row in expanded.iterrows()
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {outfile_name}"):
            idx, evidence = f.result()
            results[idx] = evidence

    # Add results to dataframe
    expanded['evidence_str'] = results

    # Parse evidence without shuffling
    parsed = [parse_evidence(text) for text in results]
    expanded['evidence_1'] = [p[0] for p in parsed]
    expanded['evidence_2'] = [p[1] for p in parsed]

    # Select and save final columns
    result_df = expanded[["ticker", "evidence_str", "evidence_1", "evidence_2", "view_1", "view_2", "buy", "sell"]]
    result_df.to_csv(outfile_name, index=False)
    print(f"Saved to {outfile_name}")

# Run batch processing
run_batch(BALANCED_CONFLICT_PROMPT_TEMPLATE, OUTPUT_PATH)