# Your AI, Not Your View: The Bias of LLMs in Investment Analysis

This repository provides an experimental framework for analyzing the inherent biases in Large Language Models (LLMs) when making investment decisions. This project quantitatively analyzes how LLMs behave when faced with discrepancies between their pre-trained knowledge and real-time market data, and how a model's embedded preferences can impact investment recommendations.

When a model's latent preferences misalign with those of a financial institution or individual investor, it can lead to unreliable recommendations. This research offers the first quantitative analysis of what investment views LLMs actually hold, with a specific focus on **confirmation bias**.

The framework works with multiple LLM providers (e.g., OpenAI, Together AI) to:

- **Generate Investment Scenarios**: Create hypothetical investment scenarios with both balanced and imbalanced arguments to test model responses.
- **Multi-Provider Support**: Works with multiple LLM providers (e.g., OpenAI, Together AI) to compare different models.
- **Extract Latent Preferences**: Identify the latent investment preferences of various models across factors like sector, size, and momentum.
- **Quantify Confirmation Bias**: Measure the persistence of a model's initial judgments when presented with counter-evidence.
- **Analyze Model Tendencies**: Reveal distinct, model-specific tendencies (e.g., a preference for large-cap stocks and contrarian strategies).

## Models Used

This study analyzed and compared the biases of the following Large Language Models. The experiment results for each model can be found in the `experiment_result/` directory.

| Model Provider | Model Name | Knowledge Cutoff |
| --- | --- | --- |
| OpenAI | GPT-4.1 | June 2024 |
| Google | Gemini 2.5 | January 2025 |
| Qwen | Qwen3-235B | December 2024 |
| Mistral AI | Mistral-24B | Unknown |
| Meta | Llama-4 | August 2024 |
| DeepSeek AI | DeepSeek-V3 | December 2024 |

## Repository Structure

```
.
├── evidence_generation_intensity.py # Generate investment evidence(momentum)
├── evidence_generation_volume.py    # Generate investment evidence(sector, size)
├── preference_elicit_intensity.py    # preference elicit(momentum)
├── preference_elicit_volume.py    # preference elicit(sector, size)
├── preference_agg.py    # preference aggregation
├── bias_verification_intensity.py    # bias_verification(sector, size)
├── bias_verification_volume.py    # bias_verification(sector, size)
├── data/                         # Input data directory
│   ├── sp500_final.csv          # S&P 500 company list
│   ├── evidence_corpus_qual.csv # Qualitative evidence
│   └── ...
├── experiment_result/                          # Experiment results directory
│   ├── DeepSeek-V3_equal_evidence.csv     # preference elicit result
│   ├── DeepSeek-V3_size_bias.csv    # most preferred group
│   └── ...
└── result/                 # Output results directory

```

## Prerequisites

### Python Dependencies

```bash
pip install pandas numpy openai google-genai together 

```

### API Keys

Set up the following environment variables with your API keys:

```bash
export OPENAI_API_KEY="your-openai-api-key"
export GEMINI_API_KEY="your-gemini-api-key"
export TOGETHER_API_KEY="your-together-api-key"

```

## Citation

If you use this code in your research, please cite:

```
@software{llm_bias_analysis,
  title = {LLM Investment Decision Bias Analysis},
  year = {2024},
  url = {<https://github.com/yourusername/llm-bias-analysis>}
}

```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or issues, please open an issue on GitHub or contact [hoyounglee@unist.ac.kr].
