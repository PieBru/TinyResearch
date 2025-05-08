## Table of Contents
- [TinyResearch 🦉](#tinyresearch-)
  - [Features](#features)
  - [Project Structure](#project-structure)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Command-Line Interface (`tinyresearch.py`)](#command-line-interface-tinyresearchpy)
      - [Provider-Specific Examples:](#provider-specific-examples)
    - [Web UI (`tinyresearch_ui.py`)](#web-ui-tinyresearch_uipy)
  - [Running Unit Tests](#running-unit-tests)
    - [Using Python's `unittest` module](#using-pythons-unittest-module)
    - [Using `pytest` (Recommended)](#using-pytest-recommended)
  - [To Do](#to-do)
  - [Credits](#credits)

# TinyResearch 🦉

[TinyResearch}(https://github.com/PieBru/TinyResearch) is a fully local, self-hostable, privacy oriented, lightweight AI agent, designed to answer complex questions by breaking them down, searching the web, visiting URLs, and synthesizing information. It is inspired by DeepResearch from jina.ai, as described by Han Xiao on his [blog post](https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch/)

From the jina.ai DeepResearch github repo:
> Keep searching, reading webpages, reasoning until an answer is found (or the token budget is exceeded). Useful for deeply investigating a query.
> [!IMPORTANT]  
> Unlike OpenAI/Gemini/Perplexity's "Deep Research", we focus solely on **finding the right answers via our iterative process**. We don't optimize for long-form articles, that's a **completely different problem** – so if you need quick, concise answers from deep search, you're in the right place. If you're looking for AI-generated long reports like OpenAI/Gemini/Perplexity does, this isn't for you.

## Features

-   Breaks down complex questions into manageable sub-questions.
-   Performs web searches using DuckDuckGo (with planned support for specialized engines like Wikipedia, Arxiv, Pubmed).
-   Visits URLs to extract textual content.
-   Utilizes Large Language Models (LLMs) via Gemini or LiteLLM (e.g., Ollama) for decision-making, summarization, and evaluation.
-   Manages a token budget to control LLM usage.
-   Includes a command-line interface (`tinyresearch.py`) and a Streamlit-based web UI (`tinyresearch_ui.py`).

## Project Structure

-   `requirements.txt`: Licalhost:8501s intentionally untracked files that Git should ignore.
-   `LICENSE`: Project's open-source license.

## Installation

1.  Clone the repository (once it's on GitHub):
    ```bash
    git clone https://github.com/PieBru/TinyResearch.git
    cd TinyResearch 
    ```
2.  Create and activate a Python virtual environment. You can use Python's built-in `venv` module or the faster `uv` tool.

    **Using `uv` (recommended for speed):**
    ```bash
    uv venv
    source .venv/bin/activate
    uv pip install -r requirements.txt -U
    ```
    **Alternatively, using standard `venv` and `pip`:**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt -U
    ```
3.  If using the `gemini` LLM provider, ensure the `GEMINI_API_KEY` environment variable is set.

## Usage

### Command-Line Interface (`tinyresearch.py`)

```bash
python tinyresearch.py --help
python tinyresearch.py "What is now the date and CET time in Rome?"
python tinyresearch.py "What are the most recent advancements in deploying AI for early cancer"
python tinyresearch.py "What are the most recent (last 6 months) advancements in deploying AI for early cancer detection, and what are the primary ethical concerns being discussed regarding these new methods?"
python tinyresearch.py "Compare the latest announced specifications, target launch windows, and primary mission objectives of the top three private space companies aiming for Mars colonization within the next two decades."
python tinyresearch.py "What are the emerging trends in sustainable urban agriculture in densely populated Asian megacities over the past year, including specific examples of successful projects and the key challenges they are overcoming?"
python tinyresearch.py "Analyze the most significant shifts in global semiconductor supply chain strategies announced by major chip manufacturers and governments in the last 9 months, and what are the projected impacts on consumer electronics availability and pricing?"
python tinyresearch.py "What are the cutting-edge developments in biodegradable plastics derived from non-food biomass sources reported in the last year, focusing on scalability and end-of-life environmental impact compared to traditional plastics?"
python tinyresearch.py "How have recent (last 12 months) breakthroughs in quantum error correction affected the projected timelines for fault-tolerant quantum computing, and which research institutions or companies are leading these specific error correction advancements?"
```

#### Provider-Specific Examples:

**1. Using Ollama:**
Make sure your Ollama server is running (usually at `http://localhost:11434`).
```bash
# Example using qwen2.5 model from Ollama
python tinyresearch.py --llm_provider litellm --llm_model ollama/qwen2.5 "Your question"

# If your Ollama endpoint is different from the default:
python tinyresearch.py --llm_provider litellm --llm_provider_endpoint http://your-ollama-host:11434 --llm_model ollama/llama3 "Your question"
```

**2. Using LM Studio (OpenAI-compatible endpoint):**
Start your LM Studio server and load a model. The OpenAI-compatible endpoint is typically `http://localhost:1234/v1`.
The model name for `--llm_model` should match what LM Studio expects (often the path or identifier shown in LM Studio).
```bash
python tinyresearch.py --llm_provider litellm --llm_provider_endpoint http://localhost:1234/v1 --llm_model local-model/ggml-model-name "Your question"
```
You might need to set an API key like `OPENAI_API_KEY=lm-studio` in your environment, though often not strictly required for local LM Studio.

**3. Using OpenRouter (Cloud, OpenAI-compatible, see https://openrouter.ai/models?q=free):**
You'll need an OpenRouter API key. Set it as an environment variable: `export OPENROUTER_API_KEY="your_openrouter_api_key"`
```bash
python tinyresearch.py --llm_provider litellm --llm_provider_endpoint https://openrouter.ai/api/v1 --llm_model openrouter/qwen/qwen3-235b-a22b:free "Your question"
```

**4. Using Cerebras (Cloud, OpenAI-compatible, see https://cloud.cerebras.ai):**
You'll need your Cerebras API endpoint and API key. Set the API key as an environment variable (e.g., `export CEREBRAS_API_KEY="your_cerebras_api_key"` or `export OPENAI_API_KEY="your_cerebras_api_key"` if it uses that convention via LiteLLM).
```bash
python tinyresearch.py --llm_provider litellm --llm_provider_endpoint https://your-cerebras-api.com/v1 --llm_model cerebras/btlm-3b-8k-base "Your question"
```

**5. Using Gemini (Native API):**
Ensure the `GEMINI_API_KEY` environment variable is set: `export GEMINI_API_KEY="your_gemini_api_key"`
```bash
python tinyresearch.py --llm_provider gemini --llm_model models/gemini-2.0-flash-latest "Your question"
# Or for Gemini Pro:
# python tinyresearch.py --llm_provider gemini --llm_model gemini-pro "Your question"
```

### Web UI (`tinyresearch_ui.py`)
A more feature-rich Streamlit UI is also available:
```bash
python tinyresearch_ui.py
```
Then open your web browser to the URL provided by Streamlit (usually `http://localhost:8501`).

## Running Unit Tests

The project includes a suite of unit tests to ensure its components function correctly. The tests are located in the `tests/` directory.

### Using Python's `unittest` module

You can run the tests directly using Python's built-in `unittest` module. Navigate to the project's root directory (`/run/media/piero/NVMe-4TB/Piero/Git/TinyResearch/`) in your terminal and execute:

```bash
python -m unittest tests.test_tinyresearch
```

Alternatively, if you are inside the `tests/` directory:
```bash
python -m unittest test_tinyresearch.py
```

### Using `pytest` (Recommended)

`pytest` is a popular testing framework that offers more features and often more readable output. If you don't have it installed, you can install it with pip:
```bash
pip install pytest
```
Once installed, navigate to the project's root directory (`/run/media/piero/NVMe-4TB/Piero/Git/TinyResearch/`) and simply run:
```bash
pytest
```
`pytest` will automatically discover and run the tests in the `tests/` directory.

## To Do

-   Add agentic logic and tools to use specialized search engines (Wikipedia, Arxiv, Pubmed, etc.).

## Credits
Huge thanks to Han Xiao from jina.ai for his inspirational [blog post](https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch/) and [follow up](https://jina.ai/news/snippet-selection-and-url-ranking-in-deepsearch-deepresearch/)
