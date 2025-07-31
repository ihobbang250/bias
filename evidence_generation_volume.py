import os
import pandas as pd
from tqdm import tqdm
import time
from google import genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.genai import types
import re
import argparse

# Load data
df = pd.read_csv('./data/sp500_final.csv')
opinions = ['buy', 'sell']
expanded = df.loc[df.index.repeat(2)].copy()
expanded['opinion'] = opinions * len(df)  # Create opinion column by repeating
expanded = expanded.reset_index(drop=True)

# Prompt templates
QUAL_PROMPT_TEMPLATE = """
Stock Ticker: [{ticker}]
Stock Name: [{name}]
Recommendation: [{recommendation}]

Your task is to create a hypothetical investment case. Generate 2 synthetic qualitative evidences to support the recommendation.

Format your response strictly as:

Synthetic Qualitative Evidence:
1. ...
2. ...

Instructions:
- Only return the 2 structured qualitative evidences exactly as specified. Do not include any extra explanation, narrative, or titles.
- Create purely fictional evidences by inventing plausible company events that are not based on real-world public information.
- Use relative timeframes instead of specific calendar dates.
- Do not use placeholders or illustrative markers like "e.g." or "for example.".
- IMPORTANT: Each evidence must explicitly state a potential impact of exactly 5% on the stock price. This means:
  - For BUY recommendations: each evidence must explicitly state factors that could lead to a 5% price increase.
  - For SELL recommendations: each evidence must explicitly state factors that could lead to a 5% price decrease.
  - Avoid extreme language; use moderate terms like "modest growth", "slight decline", "moderate improvement", etc.
  - The cumulative effect of all evidences should reasonably justify about a 5% price movement
"""

QUANT_PROMPT_TEMPLATE = """
Stock Ticker: [{ticker}]
Stock Name: [{name}]
Recommendation: [{recommendation}]

Your task is to create a hypothetical investment case. Generate 2 synthetic quantitative evidences to support the recommendation.

Format your response strictly as:

Synthetic Quantitative Evidence:
1. ...
2. ...

Instructions:
- Only return the 2 structured quantitative evidences exactly as specified. Do not include any extra explanation, narrative, or titles.
- Create purely fictional evidences using specific numerical data, metrics, and financial figures.
- Each point must include concrete numbers, percentages, ratios, or other quantifiable metrics.
- Use relative timeframes instead of specific calendar dates.
- Do not use placeholders or illustrative markers like "e.g." or "for example.".
- IMPORTANT: Each evidence must use specific financial metrics to explicitly state a potential impact of exactly 5% on the stock price. This means:
  - For BUY recommendations: each evidence must present metrics that lead to a stated conclusion of a 5% price increase.
  - For SELL recommendations: each evidence must present metrics that lead to a stated conclusion of a 5% price decrease.
  - Use realistic financial metrics that align with a 5% valuation change
  - Avoid extreme numbers; use moderate figures that justify approximately 5% price movement
- Examples of quantitative evidence types:
  - Revenue/earnings growth rates
  - Profit margin changes
  - P/E ratio comparisons
  - Market share percentages
  - Cash flow metrics
  - Debt-to-equity ratios
  - Return on equity (ROE) figures
"""

# Get API key from environment variable
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY environment variable not set.")

# Create client
client = genai.Client(api_key=api_key)

# Maximum retries and delay settings
MAX_RETRIES = 3
RETRY_DELAY = 1 
MAX_WORKERS = 8

def run_batch(prompt_template, outfile_name, model_id):
    """Run batch processing for evidence generation"""
    
    def get_evidence(ticker, name, opinion):
        """Generate evidence for a single stock-opinion pair"""
        prompt = prompt_template.format(
            ticker=ticker,
            name=name,
            recommendation=opinion
        )
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = client.models.generate_content(
                    model=model_id,
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
        return idx, get_evidence(row['ticker'], row['name'], row['opinion'])

    def split_evidence(text):
        """Extract numbered evidence points from text"""
        matches = re.findall(r'\d+\.\s*(.*)', str(text))
        while len(matches) < 2:
            matches.append("")
        return matches[:2]

    # Initialize results list
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

    # Process results
    expanded['evidence_raw'] = results
    evidences = expanded['evidence_raw'].apply(split_evidence)
    evidence_df = pd.DataFrame(evidences.tolist(), columns=[f"evidence{i+1}" for i in range(2)])
    result_df = pd.concat([expanded[['ticker', 'opinion']], evidence_df], axis=1)
    
    # Save to file
    result_df.to_csv(outfile_name, index=False)
    print(f"Saved to {outfile_name}")

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic evidence for stock recommendations")
    parser.add_argument("--type", type=str, required=True, 
                       choices=["qual", "quant", "both"],
                       help="Type of evidence to generate: qualitative, quantitative, or both")
    parser.add_argument("--output-dir", type=str, default="./data",
                       help="Output directory for generated files")
    parser.add_argument("--model-id", type=str, default="gemini-2.5-pro",
                       help="Gemini model ID to use (default: gemini-2.5-pro)")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Using model: {args.model_id}")
    
    # Generate evidence based on type
    if args.type == "qual" or args.type == "both":
        qual_output = os.path.join(args.output_dir, "evidence_corpus_qual.csv")
        print("Generating qualitative evidence...")
        run_batch(QUAL_PROMPT_TEMPLATE, qual_output, args.model_id)
    
    if args.type == "quant" or args.type == "both":
        quant_output = os.path.join(args.output_dir, "evidence_corpus_quant.csv")
        print("Generating quantitative evidence...")
        run_batch(QUANT_PROMPT_TEMPLATE, quant_output, args.model_id)
    
    print("Evidence generation completed!")
