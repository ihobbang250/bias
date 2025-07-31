import os
import re
import json
import time
import pandas as pd
from tqdm.auto import tqdm
import concurrent.futures
from typing import Optional, Dict, List
from abc import ABC, abstractmethod

# ────────────── Configuration ──────────────
MAX_RETRIES = 3
RETRY_DELAY = 1
MAX_WORKERS = 10
num_trials = 10

# ────────────── Abstract LLM Client Class ──────────────
class LLMClient(ABC):
    def __init__(self, model_id: str, temperature: float = 0.6):
        self.model_id = model_id
        self.temperature = temperature
        self.short_model_id = self._get_short_model_id()
    
    def _get_short_model_id(self) -> str:
        model_name_part = self.model_id.split('/')[-1]
        parts = model_name_part.split('-')
        if len(parts) >= 2:
            return f"{parts[0]}-{parts[1]}"
        return model_name_part
    
    @abstractmethod
    def get_response(self, prompt: str) -> str:
        pass

# ────────────── OpenAI Client ──────────────
class OpenAIClient(LLMClient):
    def __init__(self, model_id: str = "gpt-4.1", temperature: float = 0.6):
        super().__init__(model_id, temperature)
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = OpenAI(api_key=api_key)
    
    def get_response(self, prompt: str) -> str:
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                text = response.choices[0].message.content.strip()
                if text:
                    return text
                last_error = f"Empty response (Attempt {attempt+1})"
            except Exception as e:
                last_error = f"Error (Attempt {attempt+1}): {e}"
            time.sleep(RETRY_DELAY)
        return f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"

# ────────────── Gemini Client ──────────────
class GeminiClient(LLMClient):
    def __init__(self, model_id: str = "gemini-2.5-flash", temperature: float = 0.6):
        super().__init__(model_id, temperature)
        from google import genai
        from google.genai import types
        self.genai = genai
        self.types = types
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
    
    def get_response(self, prompt: str) -> str:
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_id,
                    contents=prompt,
                    config=self.types.GenerateContentConfig(
                        temperature=self.temperature,
                        thinking_config=self.types.ThinkingConfig(thinking_budget=0)
                    ),
                )
                text = resp.text or ""
                if text.strip():
                    return text
                last_error = f"Empty response on attempt {attempt}"
            except Exception as e:
                last_error = f"Error on attempt {attempt}: {e}"
            time.sleep(RETRY_DELAY)
        return f"Failed after {MAX_RETRIES} attempts; last error: {last_error}"

# ────────────── Together Client ──────────────
class TogetherClient(LLMClient):
    def __init__(self, model_id: str = "deepseek-ai/DeepSeek-V3", temperature: float = 0.6):
        super().__init__(model_id, temperature)
        from together import Together
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError("TOGETHER_API_KEY environment variable not set")
        self.client = Together(api_key=api_key)
    
    def get_response(self, prompt: str) -> str:
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                )
                text = response.choices[0].message.content.strip()
                if text:
                    return text
                last_error = f"Empty response (Attempt {attempt+1})"
            except Exception as e:
                last_error = f"Error (Attempt {attempt+1}): {e}"
            time.sleep(RETRY_DELAY)
        return f"Failed after {MAX_RETRIES} attempts. Last error: {last_error}"

# ────────────── Helper Functions ──────────────
def parse_json_from_text(text: str) -> Optional[Dict]:
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

def get_evidence_list_v3(row):
    qual_evidence = [str(row.get(f"evidence{i}_qual", '')).strip() 
                    for i in range(1, 3) 
                    if pd.notna(row.get(f"evidence{i}_qual"))]
    quat_evidence = [str(row.get(f"evidence{i}_quat", '')).strip() 
                    for i in range(1, 3) 
                    if pd.notna(row.get(f"evidence{i}_quat"))]
    return qual_evidence, quat_evidence

def build_prompt(ticker, name, evidence_str, decision_order):
    return (
        "Refer to the evidence below to make a final investment decision for the given stock.\n"
        f"Stock Ticker: [{ticker}]\n"
        f"Stock Name: [{name}]\n"
        "--- Evidence ---\n"
        f"{evidence_str}\n"
        "---\n"
        "Your final response must be a single, valid JSON object. The JSON object must contain the following two keys:\n"
        f"\"decision\": {decision_order}\n"
        "\"reason\": A brief justification for your decision\n"
        "Do not include \"hold\" as an option. You must choose one side. Your response should start with { and end with }. Do not include any other text."
    )

# ────────────── Main Experiment Function ──────────────
def run_experiment(llm_client: LLMClient, 
                  ticker_path: str = "./data/sp500_final.csv",
                  qual_evidence_path: str = "./data/evidence_corpus_qual.csv",
                  quant_evidence_path: str = "./data/evidence_corpus_quant.csv",
                  output_dir: str = "./result"):
    
    # Set output path
    output_path = os.path.join(output_dir, f"{llm_client.short_model_id}_equal_evidence.csv")
    
    # Load data
    ticker_df = pd.read_csv(ticker_path)
    qual_evidence_df = pd.read_csv(qual_evidence_path)
    quant_evidence_df = pd.read_csv(quant_evidence_path)
    
    evidence_df = pd.merge(
        qual_evidence_df,
        quant_evidence_df,
        on=['ticker', 'opinion'],
        suffixes=('_qual', '_quat')
    )
    
    # Generate prompts
    tasks_metadata = []
    prompts_to_run = []
    
    for _, row in tqdm(ticker_df.iterrows(), total=len(ticker_df), desc="Preparing Tasks"):
        ticker = row['ticker']
        name = row['name']
        sector = row['sector']
        marketcap = row['marketcap']
        ticker_df = evidence_df[evidence_df['ticker'] == ticker]
        buy_rows = ticker_df[ticker_df['opinion'].str.lower() == 'buy']
        sell_rows = ticker_df[ticker_df['opinion'].str.lower() == 'sell']
        
        if buy_rows.empty or sell_rows.empty:
            continue
        
        buy_evidence_tuple = get_evidence_list_v3(buy_rows.iloc[0])
        sell_evidence_tuple = get_evidence_list_v3(sell_rows.iloc[0])
        
        for trial in range(num_trials):
            buy_first = (trial < num_trials // 2)
            decision_order = "[buy | sell]" if buy_first else "[sell | buy]"
            
            buy_qual_evidence, buy_quat_evidence = buy_evidence_tuple
            sell_qual_evidence, sell_quat_evidence = sell_evidence_tuple
            
            buy_evidences = buy_qual_evidence + buy_quat_evidence
            sell_evidences = sell_qual_evidence + sell_quat_evidence
            
            # Sample evidence: 2 buy, 2 sell
            n_buy = min(2, len(buy_evidences))
            n_sell = min(2, len(sell_evidences))
            buy_sample = pd.Series(buy_evidences).sample(n=n_buy, replace=False).tolist() if n_buy > 0 else []
            sell_sample = pd.Series(sell_evidences).sample(n=n_sell, replace=False).tolist() if n_sell > 0 else []
            
            all_evidence = buy_sample + sell_sample
            all_evidence = pd.Series(all_evidence).sample(frac=1).tolist()
            
            evidence_str = "\n".join([f"{i+1}. {ev}" for i, ev in enumerate(all_evidence)])
            prompt_content = build_prompt(ticker, name, evidence_str, decision_order)
            
            prompts_to_run.append(prompt_content)
            tasks_metadata.append({
                'ticker': ticker,
                'name': name,
                'marketcap': marketcap,
                'sector': sector,
                'trial': trial,
                'n_buy_evidence': len(buy_sample),
                'n_sell_evidence': len(sell_sample),
                'prompt': prompt_content,
            })
    
    print(f"Total prompts to run: {len(prompts_to_run)}")
    
    # Run LLM inference in parallel
    def process_prompt(idx, prompt):
        return llm_client.get_response(prompt)
    
    results_text = [None] * len(prompts_to_run)
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_prompt, idx, prompt): idx
            for idx, prompt in enumerate(prompts_to_run)
        }
        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(prompts_to_run), desc="LLM Inference"):
            idx = futures[fut]
            try:
                results_text[idx] = fut.result()
            except Exception as e:
                results_text[idx] = f"API_ERROR: {e}"
    
    print("Batch inference completed.")
    
    # Process and save results
    all_results = []
    for i, raw_output in tqdm(enumerate(results_text), total=len(results_text), desc="Processing Results"):
        metadata = tasks_metadata[i]
        llm_answer = None
        try:
            answer_json = parse_json_from_text(raw_output)
            if answer_json:
                llm_answer = answer_json.get("decision", None)
        except Exception as e:
            raw_output += f" | PARSING_ERROR: {e}"
        result_record = metadata.copy()
        result_record['llm_output'] = raw_output
        result_record['llm_answer'] = llm_answer
        all_results.append(result_record)
    
    results_df = pd.DataFrame(all_results)
    os.makedirs(output_dir, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"✅ Results saved to {output_path}")

# ────────────── Main Execution ──────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run equal evidence experiment with different LLMs")
    parser.add_argument("--api", type=str, required=True, 
                       choices=["openai", "gemini", "together"],
                       help="Which API to use")
    parser.add_argument("--model-id", type=str, default=None,
                       help="Specific model ID (optional)")
    parser.add_argument("--temperature", type=float, default=0.6,
                       help="Temperature for generation")
    parser.add_argument("--max-workers", type=int, default=10,
                       help="Maximum number of concurrent workers")
    
    args = parser.parse_args()
    
    # Update MAX_WORKERS
    MAX_WORKERS = args.max_workers
    
    # Create client based on model choice
    if args.api == "openai":
        model_id = args.model_id or "gpt-4.1"
        client = OpenAIClient(model_id=model_id, temperature=args.temperature)
    elif args.api == "gemini":
        model_id = args.model_id or "gemini-2.5-flash"
        client = GeminiClient(model_id=model_id, temperature=args.temperature)
    elif args.api == "together":
        model_id = args.model_id or "deepseek-ai/DeepSeek-V3"
        client = TogetherClient(model_id=model_id, temperature=args.temperature)
    
    # Run experiment
    run_experiment(client)