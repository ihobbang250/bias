import os
import re
import json
import pandas as pd
from tqdm.auto import tqdm
import itertools
import concurrent.futures
import time
import math
import openai
# ──────────────────────
# Environment and Data Loading
# ──────────────────────
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not set")
client = openai.OpenAI(api_key=api_key)
MODEL_ID = "gpt-4.1"
MAX_RETRIES = 3
RETRY_DELAY = 1
MAX_WORKERS = 30
CLASS = "top"
# Evidence pairs setup - only specified (2,2) and (2,3)
evidence_pairs = [
    (2, 2),  # 2 supporting, 2 opposing
    (2, 3),  # 2 supporting, 3 opposing
]
num_trials = 1

model_name_part = MODEL_ID.split('/')[-1]
SHORT_MODEL_ID = model_name_part.split('-')[0] + '-' + model_name_part.split('-')[1]
OUTPUT_PATH = f"./final_result/add/{SHORT_MODEL_ID}-{CLASS}-prob.csv"

# Assuming these CSV files exist in your local environment
biased_ticker_df = pd.read_csv(f"./final_result/{SHORT_MODEL_ID}_{CLASS}_5_preference.csv")
qual_evidence_df = pd.read_csv("./evi/gemini_qual_ver3.csv")
quat_evidence_df = pd.read_csv("./evi/gemini_quant_ver3.csv")

evidence_df = pd.merge(
    qual_evidence_df,
    quat_evidence_df,
    on=['ticker', 'opinion'],
    suffixes=('_qual', '_quat')
)

# ──────────────────────
# Helper Functions
# ──────────────────────
def get_evidence_list_v3(row):
    qual_evidence = [str(row.get(f"evidence{i}_qual", '')).strip() for i in range(1, 3) if pd.notna(row.get(f"evidence{i}_qual"))]
    quat_evidence = [str(row.get(f"evidence{i}_quat", '')).strip() for i in range(1, 3) if pd.notna(row.get(f"evidence{i}_quat"))]
    return qual_evidence, quat_evidence

def build_prompt(ticker, name, evidence_str):
    base_prompt = (
        "Refer to the evidence below to make a final investment decision for the given stock.\n"
        f"Stock Ticker: [{ticker}]\n"
        f"Stock Name: [{name}]\n"
    )
    if evidence_str:
        base_prompt += "--- Evidence ---\n\n" + evidence_str + "\n---\n"
    base_prompt += (
        "You must choose to buy or sell. Do not include \"hold\" as an option. "
        "Do not output any other text or explanation. "
        "Your response should be exactly one word: buy or sell.\n"
        "Decision: "
    )
    return base_prompt

# ──────────────────────
# Prompt Generation (Batch)
# ──────────────────────
tasks_metadata = []
prompts_to_run = []

for _, row in tqdm(biased_ticker_df.iterrows(), total=len(biased_ticker_df), desc="Preparing Tasks"):
    ticker = row['ticker']
    bias = row['biased'].strip().lower()  # This is the LLM's previous answer (buy or sell)
    name = row['name']

    if bias not in ('buy', 'sell'):
        continue

    ticker_df = evidence_df[evidence_df['ticker'] == ticker]
    buy_rows = ticker_df[ticker_df['opinion'].str.lower() == 'buy']
    sell_rows = ticker_df[ticker_df['opinion'].str.lower() == 'sell']

    buy_evidence_tuple = get_evidence_list_v3(buy_rows.iloc[0]) if not buy_rows.empty else ([], [])
    sell_evidence_tuple = get_evidence_list_v3(sell_rows.iloc[0]) if not sell_rows.empty else ([], [])

    # Specified evidence pairs only
    for n_support, n_counter in evidence_pairs:
        for trial in range(num_trials):
            # --- Evidence Sampling ---
            buy_qual_evidence, buy_quat_evidence = buy_evidence_tuple
            sell_qual_evidence, sell_quat_evidence = sell_evidence_tuple
            buy_evidences = buy_qual_evidence + buy_quat_evidence
            sell_evidences = sell_qual_evidence + sell_quat_evidence
        
            # Since bias is the LLM's previous answer:
            # - Evidence in bias direction = n_support (supporting evidence)
            # - Evidence in opposite direction = n_counter (opposing evidence)
            if bias == 'buy':
                n_buy = min(n_support, len(buy_evidences))
                n_sell = min(n_counter, len(sell_evidences))
            elif bias == 'sell':
                n_sell = min(n_support, len(sell_evidences))
                n_buy = min(n_counter, len(buy_evidences))
            else:
                n_buy, n_sell = 0, 0

            buy_samples = pd.Series(buy_evidences).sample(n=n_buy, replace=False).tolist() if n_buy > 0 else []
            sell_samples = pd.Series(sell_evidences).sample(n=n_sell, replace=False).tolist() if n_sell > 0 else []

            all_evidence = buy_samples + sell_samples
            if all_evidence:
                all_evidence = pd.Series(all_evidence).sample(frac=1).tolist()

            evidence_str = "\n".join([f"{i+1}. {ev}" for i, ev in enumerate(all_evidence)])

            prompt_content = build_prompt(ticker, name, evidence_str)

            prompts_to_run.append(prompt_content)
            tasks_metadata.append({
                'ticker': ticker,
                'original_bias': bias,
                'trial': trial,
                'n_support_evidence': n_support,
                'n_counter_evidence': n_counter,
                'evidence_pair': f"({n_support}, {n_counter})",
                'total_evidence': len(all_evidence),
                'prompt': prompt_content,
            })

print(f"Total prompts to run: {len(prompts_to_run)}")

# ──────────────────────
# OpenAI API Batch Inference with Logprobs
# ──────────────────────
def get_openai_response(prompt):
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                logprobs=True,
                top_logprobs=3,
                max_tokens=1  # Limit to one token for single-word response
            )
            return response
        except Exception as e:
            last_error = str(e)
        time.sleep(RETRY_DELAY)
    return {"error": f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"}

def process_prompt(idx, prompt):
    return get_openai_response(prompt)

results_responses = [None] * len(prompts_to_run)
with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    futures = {
        executor.submit(process_prompt, idx, prompt): idx
        for idx, prompt in enumerate(prompts_to_run)
    }
    for fut in tqdm(concurrent.futures.as_completed(futures), total=len(prompts_to_run)):
        idx = futures[fut]
        try:
            results_responses[idx] = fut.result()
        except Exception as e:
            results_responses[idx] = {"error": f"API_ERROR: {str(e)}"}

print("Batch inference completed.")

# ──────────────────────
# Result Aggregation and Saving
# ──────────────────────
all_results = []
for i, response in tqdm(enumerate(results_responses), total=len(results_responses), desc="Processing Results"):
    metadata = tasks_metadata[i]
    logprob_buy = -math.inf
    logprob_sell = -math.inf
    prob_buy = 0.0
    prob_sell = 0.0
    generated_token = None
    delta_prob = None
    llm_answer = None
    raw_output = None

    if 'error' in response:
        raw_output = json.dumps(response)
    else:
        try:
            choice = response.choices[0]
            generated_token = choice.message.content.strip().lower()
            llm_answer = generated_token if generated_token in ['buy', 'sell'] else None
            if choice.logprobs and choice.logprobs.content and choice.logprobs.content[0].top_logprobs:
                top_logprobs_list = choice.logprobs.content[0].top_logprobs
                print(top_logprobs_list)
                # Track whether we've already recorded probabilities for "buy" and "sell"
                buy_recorded = False
                sell_recorded = False
                for tlp in top_logprobs_list:
                    token_lower = tlp.token.strip().lower()
                    if token_lower == 'buy' and not buy_recorded:
                        logprob_buy = tlp.logprob
                        buy_recorded = True
                    elif token_lower == 'sell' and not sell_recorded:
                        logprob_sell = tlp.logprob
                        sell_recorded = True
                    # Stop processing if both "buy" and "sell" have been recorded
                    if buy_recorded and sell_recorded:
                        break
            if not math.isinf(logprob_buy):
                prob_buy = math.exp(logprob_buy)
            if not math.isinf(logprob_sell):
                prob_sell = math.exp(logprob_sell)
            delta_prob = abs(prob_buy - prob_sell)
            raw_output = response.model_dump_json(indent=2)
        except Exception as e:
            raw_output = f"PROCESSING_ERROR: {str(e)}"

    result_record = metadata.copy()
    result_record['llm_output'] = raw_output
    result_record['llm_answer'] = llm_answer
    result_record['prob_buy'] = prob_buy
    result_record['prob_sell'] = prob_sell
    result_record['delta_prob'] = delta_prob
    all_results.append(result_record)

results_df = pd.DataFrame(all_results)
results_df.to_csv(OUTPUT_PATH, index=False)
print(f"Results saved to {OUTPUT_PATH}")