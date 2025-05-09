#!/usr/bin/env python3

# TinyResearch - See LICENSE and original project: https://github.com/PieBru/TinyResearch
# Credits: Huge thanks to Han Xiao for his blog post and follow up:
# - https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch/
# - https://jina.ai/news/snippet-selection-and-url-ranking-in-deepsearch-deepresearch/

# Versioning scheme: YYYY.MM.DD.progressive_letter (e.g., 2024.05.10.a)
__version__ = "2024.05.10.c"

import sys
import time
import random
import json
import re
import warnings
import os
import argparse
import sys

# Suppress the specific UserWarning from protobuf about gencode versions
warnings.filterwarnings("ignore", category=UserWarning, message=".*Protobuf gencode version.*older than the runtime version.*")

# Torch is not a direct dependency for core logic anymore if sentence-transformers is removed.
# FAISS or other libraries might still use it.
try:
    from rich import print
except ImportError:
    print("Missing `rich` library. To install: pip install rich" )
    # Don't exit(), the `rich` library is only an aesthetical enhancement.

try:
    import google.generativeai as genai
    import litellm
    # os.environ['LITELLM_LOG'] = 'DEBUG'
    from duckduckgo_search import DDGS # Import DDGS here
    import requests
    from bs4 import BeautifulSoup
    from datetime import datetime # For current time
    import pytz # For timezone handling
    import yfinance as yf # For stock data
    import faiss # For vector database
    import ollama # Import the ollama library
    import numpy as np # For FAISS
except ImportError:
    print("Missing required libraries. Please install them: pip install -r requirements.txt -U")
    # exit() # Avoid calling exit() in a module
    raise # Re-raise the ImportError so the caller (pytest or streamlit) knows about it


# --- Constants (defaults can be overridden by CLI args) ---
DEFAULT_TOKEN_BUDGET = 30000
DEFAULT_MAX_SIMULATION_STEPS = 20
DEFAULT_TIME_BUDGET_MINUTES = 10 # New time budget constant
DEFAULT_MAX_URL_CONTENT_LENGTH = 2000
DEFAULT_OUTPUT_FORMAT = "text"

DEFAULT_LLM_PROVIDER = "litellm" # See https://docs.litellm.ai/docs/providers

# Ollama, see https://docs.litellm.ai/docs/providers/ollama
#DEFAULT_LLM_PROVIDER_ENDPOINT = "http://localhost:11434"
#DEFAULT_LLM_MODEL = "ollama/qwen2.5" # Also "ollama/qwen3" and others that are tools-enabled
#DEFAULT_LITELLM_EMBEDDING_MODEL = "ollama/bge-m3"

# LM Studio, see https://docs.litellm.ai/docs/providers/lm_studio
DEFAULT_LLM_PROVIDER_ENDPOINT = "http://localhost:1234/v1"
DEFAULT_LLM_MODEL = "lm_studio/qwen2.5-7b-instruct-1m"
DEFAULT_LITELLM_EMBEDDING_MODEL = "lm_studio/text-embedding-bge-m3"

VECTOR_DB_SEARCH_TOP_N = 3 # Number of similar items to retrieve from vector DB
CHROME_WEB_BROWSER_USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/112.0.5615.49 Safari/537.36'
DEFAULT_WEB_BROWSER_USER_AGENT = CHROME_WEB_BROWSER_USER_AGENT

MAX_KNOWLEDGE_SNIPPETS_IN_PROMPT = 5
MAX_URLS_TO_VISIT_IN_PROMPT = 5
MAX_SUB_QUESTIONS_PER_REFLECT = 2
MAX_SEARCH_RESULTS = 10
MAX_SEARCH_SNIPPETS_TO_STORE = 10
MAX_URLS_PER_VISIT_ACTION = 2
REQUESTS_TIMEOUT = 10
LLM_ERROR_PENALTY_TOKENS = 100
LLM_MAX_OUTPUT_TOKENS = 4096 # For agent action decisions
LLM_SUMMARIZER_MAX_OUTPUT_TOKENS = 4096 # For final summary
LLM_TEMPERATURE = 0.1
LLM_TOP_P = 0.1 # Default Top P (Nucleus Sampling)
LLM_TOP_K = 40 # Default Top K
LLM_REPEAT_PENALTY = 1.0 # Default Repeat Penalty (1.0 = no penalty)
LLM_MIN_P = 0.0 # Default Min P (often 0 means not used)
LOOP_PAUSE_SECONDS = 2

# --- Global LLM Configuration (set by main) ---
g_llm_provider = DEFAULT_LLM_PROVIDER
g_llm_provider_endpoint = DEFAULT_LLM_PROVIDER_ENDPOINT
g_llm_model = DEFAULT_LLM_MODEL
g_gemini_api_key = None

# --- Embedding and Vector DB Functions ---

def _embed_texts_litellm(texts_to_embed, context):
    """Helper function to get embeddings using LiteLLM."""
    embedding_model_name_full = context["embedding_model_name"]

    if embedding_model_name_full.startswith("ollama/"):
        ollama_model_name = embedding_model_name_full.split('/', 1)[1]
        # Use g_llm_provider_endpoint if set, otherwise default Ollama host
        ollama_host = g_llm_provider_endpoint if g_llm_provider_endpoint and "ollama" in g_llm_provider_endpoint.lower() else "http://localhost:11434"
        print(f"Requesting embeddings for {len(texts_to_embed)} text(s) using Ollama direct: model='{ollama_model_name}', host='{ollama_host}'")
        try:
            client = ollama.Client(host=ollama_host)
            all_embeddings = []
            for text_chunk in texts_to_embed: # Ollama library typically embeds one prompt at a time
                response = client.embeddings(model=ollama_model_name, prompt=text_chunk)
                all_embeddings.append(response["embedding"])
            # Token counting for Ollama direct is not straightforward from its response.
            # We can estimate based on input text.
            estimated_tokens = sum(len(text.split()) for text in texts_to_embed) # Rough estimate
            context["tokens_used"] += estimated_tokens
            print(f"Ollama direct embedding estimated input tokens: {estimated_tokens}")
            return np.array(all_embeddings, dtype=np.float32)
        except Exception as e:
            import traceback
            print(f"ERROR during Ollama direct embedding call for model '{ollama_model_name}': {type(e).__name__} - {e}")
            print(traceback.format_exc()) # Print full traceback for the direct ollama call error
            context["tokens_used"] += LLM_ERROR_PENALTY_TOKENS
            return None
    else: # Use LiteLLM for other providers
        api_base_to_use = None
        api_key_to_use = None # Initialize api_key_to_use

        if g_llm_provider_endpoint and not any(embedding_model_name_full.startswith(p) for p in ["gemini/", "openai/", "azure/", "bedrock/"]):
            api_base_to_use = g_llm_provider_endpoint
            # If this custom endpoint is for LM Studio (heuristic check), provide a dummy key
            if "1234" in g_llm_provider_endpoint or "lmstudio" in g_llm_provider_endpoint.lower(): # Heuristic for LM Studio
                api_key_to_use = "dummy_key_for_lm_studio_embeddings"

        print(f"Requesting embeddings for {len(texts_to_embed)} text(s) using LiteLLM model: {embedding_model_name_full} (API base: {api_base_to_use or 'Default'})")
        try:
            response = litellm.embedding(
                model=embedding_model_name_full,
                input=texts_to_embed,
                api_base=api_base_to_use,
                api_key=api_key_to_use # Pass the api_key
            )
            # Adjust for different response structures.
            if embedding_model_name_full.startswith("lm_studio/") and response.data and isinstance(response.data[0], dict) and "embedding" in response.data[0]:
                embeddings = [item["embedding"] for item in response.data] # For raw LM Studio dict response
            else:
                embeddings = [item.embedding for item in response.data] # Standard LiteLLM EmbeddingResponse
            
            if hasattr(response, 'usage') and response.usage:
                tokens_consumed = response.usage.total_tokens if hasattr(response.usage, 'total_tokens') else response.usage.prompt_tokens
                if tokens_consumed:
                    context["tokens_used"] += tokens_consumed
            else: # Fallback token estimation for embeddings
                estimated_tokens = sum(len(text.split()) for text in texts_to_embed) # Rough estimate
                context["tokens_used"] += estimated_tokens
                print(f"LiteLLM embedding usage not available, estimated tokens: {estimated_tokens}")

            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            print(f"Error during LiteLLM embedding call for model '{embedding_model_name_full}': {e}")
            context["tokens_used"] += LLM_ERROR_PENALTY_TOKENS 
            return None

def _get_embedding_dimension(context):
    """Determines embedding dimension by making a test call."""
    if 'embedding_dim' not in context: # Calculate and store if not already done
        print(f"Determining embedding dimension for model: {context['embedding_model_name']}...")
        dummy_embeddings_array = _embed_texts_litellm(["get dimension test string"], context)
        if dummy_embeddings_array is not None and dummy_embeddings_array.ndim == 2 and dummy_embeddings_array.shape[0] > 0:
            context['embedding_dim'] = dummy_embeddings_array.shape[1]
            print(f"Determined embedding dimension: {context['embedding_dim']}")
        else:
            raise ValueError(f"Could not determine embedding dimension using LiteLLM for model {context['embedding_model_name']}.")
    return context['embedding_dim']

def _initialize_vector_db(context):
    embedding_dim = _get_embedding_dimension(context) # Get dimension using LiteLLM
    context['vector_db_index'] = faiss.IndexFlatL2(embedding_dim)
    context['vector_db_texts'] = [] # Stores original texts, index corresponds to FAISS ID
    print(f"In-memory FAISS vector DB initialized with dimension {embedding_dim}.")

def add_text_to_vector_db(context, text_content, source_info=""):
    if not text_content or not isinstance(text_content, str) or not text_content.strip():
        return
    if 'vector_db_index' not in context:
        _initialize_vector_db(context) # Initialize if not already
        # print("Vector DB not initialized. Skipping add.") # Should be initialized now
        # return
        
    embeddings_array = _embed_texts_litellm([text_content.strip()], context)
    
    if embeddings_array is not None and embeddings_array.ndim == 2 and embeddings_array.shape[0] > 0:
        # embeddings_array is already np.float32 from _embed_texts_litellm
        context['vector_db_index'].add(embeddings_array)
        context['vector_db_texts'].append({"text": text_content.strip(), "source": source_info})
        # print(f"Added to Vector DB (ID {context['vector_db_index'].ntotal - 1}): {text_content[:100]}... (Source: {source_info})")
    else:
        print(f"Failed to get embedding for text: {text_content[:100]}...")

def query_vector_db(context, query_text, top_n=VECTOR_DB_SEARCH_TOP_N):
    if 'vector_db_index' not in context or context['vector_db_index'].ntotal == 0:
        print("Vector DB is empty or not initialized. No query performed.")
        return "No items in vector DB to search." # Return the descriptive message here
    if not query_text or not isinstance(query_text, str) or not query_text.strip():
        print("Empty query text for vector DB. No query performed.")
        return ""

    print(f"Querying Vector DB with: '{query_text[:100]}...' (top_n={top_n})")
    query_embedding_array = _embed_texts_litellm([query_text.strip()], context)

    if query_embedding_array is None or query_embedding_array.shape[0] == 0:
        print(f"Failed to get embedding for query: {query_text[:100]}...")
        return "Failed to generate query embedding."

    actual_top_n = min(top_n, context['vector_db_index'].ntotal)
    if actual_top_n == 0: return "No items in vector DB to search." # This case should ideally not be hit if ntotal > 0

    distances, indices = context['vector_db_index'].search(query_embedding_array, actual_top_n)
    
    retrieved_texts = []
    for i in indices[0]: 
        if i != -1: 
            retrieved_texts.append(f"- Source: {context['vector_db_texts'][i]['source']}, Content: {context['vector_db_texts'][i]['text'][:150]}...")
    return "\n".join(retrieved_texts) if retrieved_texts else "No relevant information found in session memory."

# --- LLM Interaction Functions (Replaces SmolAgent functionality) ---
def _make_llm_api_call(agent_name, messages, # Changed from system_prompt, full_user_prompt_text
                       llm_provider, llm_model, llm_provider_endpoint, 
                       max_output_tokens, context_for_token_counting):
    raw_response_text = ""
    print(f"Agent '{agent_name}' calling LLM ({llm_provider}/{llm_model})...")

    # Extract system prompt and user prompt text for token estimation if needed, and for Gemini
    system_prompt_for_estimation = messages[0]['content'] if messages and messages[0]['role'] == 'system' else ""
    user_prompt_for_estimation = messages[-1]['content'] if messages and messages[-1]['role'] == 'user' else ""

    if llm_provider == "gemini":
        if not g_gemini_api_key:
            print("ERROR: Gemini API key not configured for 'gemini' provider.")
            return None, "Gemini API key not configured"
        try:
            # LiteLLM handles message conversion for Gemini, but if calling Gemini directly:
            # Convert messages to Gemini's format (history + current prompt)
            # For simplicity, we'll let LiteLLM handle this if it can, or adjust if direct Gemini SDK use is strict
            # For now, assuming LiteLLM handles the messages list for Gemini provider.
            # If not, this part needs specific Gemini history/prompt construction.
            # Let's assume LiteLLM handles it for now. If it fails, we'll adjust.
            model_instance = genai.GenerativeModel(llm_model)
            gen_config_params = {
                "max_output_tokens": max_output_tokens,
                "temperature": LLM_TEMPERATURE,
                "top_p": LLM_TOP_P
            }
            if LLM_TOP_K > 0: # Gemini's top_k is an int, 0 might be invalid or mean "not used"
                gen_config_params["top_k"] = LLM_TOP_K
            # Note: Gemini API doesn't directly expose repeat_penalty or min_p in GenerationConfig
            response = model_instance.generate_content(
                # Construct content from messages. For basic, use last user message.
                # For multi-turn, Gemini SDK needs a list of Content objects.
                [part for message in messages for part in genai.types.ContentDict(role=message['role'], parts=[message['content']]).parts], # More robust
                generation_config=genai.types.GenerationConfig(**gen_config_params)
            )
            if response.usage_metadata:
                tokens = response.usage_metadata.total_token_count
                context_for_token_counting["tokens_used"] += tokens
            else: # Fallback token estimation
                estimated_tokens = sum(len(m['content'].split()) for m in messages) + len(response.text.split())
                context_for_token_counting["tokens_used"] += estimated_tokens
            raw_response_text = response.text
        except Exception as e:
            print(f"Error during Gemini API call for agent '{agent_name}': {e}")
            context_for_token_counting["tokens_used"] += LLM_ERROR_PENALTY_TOKENS
            return None, str(e)

    elif llm_provider == "litellm":
        try:
            litellm_kwargs = {
                "model": llm_model,
                "messages": messages, # Use the passed messages list
                "max_tokens": max_output_tokens,
                "temperature": LLM_TEMPERATURE,
                "top_p": LLM_TOP_P,
                # LiteLLM passes these through if the model supports them
                "frequency_penalty": LLM_REPEAT_PENALTY, # Using frequency_penalty for repeat_penalty
                # "presence_penalty": LLM_PRESENCE_PENALTY, # Could also be an option
            }
            if LLM_TOP_K > 0:
                litellm_kwargs["top_k"] = LLM_TOP_K
            if LLM_MIN_P > 0.0: # Only add if non-zero, as support varies
                litellm_kwargs["min_p"] = LLM_MIN_P # Note: Not a standard LiteLLM param, model-dependent

            model_to_send = llm_model
            if llm_model.startswith("ollama/") or (llm_provider_endpoint): # Check if endpoint is needed
                litellm_kwargs["api_base"] = llm_provider_endpoint
                # If it's an LM Studio model (identified by "lm_studio/" prefix in our config)
                # and we're providing an api_base, send the raw model name
                # and tell LiteLLM to treat it as a generic OpenAI-compatible endpoint.
                if llm_model.startswith("lm_studio/"):
                    model_to_send = llm_model.split('/', 1)[1]
                    # Prepend "openai/" to hint LiteLLM for OpenAI-compatible custom server
                    model_to_send = f"openai/{model_to_send}"
                    # OpenAI-compatible local servers often need a dummy api_key
                    litellm_kwargs["api_key"] = "dummy_key_for_lm_studio" 

            litellm_kwargs["model"] = model_to_send # Use the potentially modified model name

            # print(f"DEBUG: LiteLLM completion called by '{agent_name}' with kwargs:")
            # print(f"DEBUG: {litellm_kwargs}")
            # sys.stdout.flush() # Ensure it's flushed if redirected
            response = litellm.completion(**litellm_kwargs)
            raw_response_text = response.choices[0].message.content
            
            if hasattr(response, 'usage') and response.usage and hasattr(response.usage, 'total_tokens') and response.usage.total_tokens is not None:
                tokens = response.usage.total_tokens
                context_for_token_counting["tokens_used"] += tokens
            else: # Fallback token estimation
                estimated_tokens = sum(len(m['content'].split()) for m in messages) + len(raw_response_text.split())
                context_for_token_counting["tokens_used"] += estimated_tokens
        except Exception as e:
            print(f"Error during LiteLLM API call for agent '{agent_name}': {e}")
            context_for_token_counting["tokens_used"] += LLM_ERROR_PENALTY_TOKENS
            return None, str(e)
    else:
        return None, f"Unknown LLM provider: {llm_provider}"
    
    print(f"Agent '{agent_name}' raw LLM response:\n---\n{raw_response_text}\n---")
    return raw_response_text, None

def decide_json_action(agent_name, messages, # Changed from system_prompt, user_prompt_text
                       llm_provider, llm_model, llm_provider_endpoint, 
                       max_output_tokens, context_for_token_counting):
    # Construct messages list if system_prompt and user_prompt_text are passed for backward compatibility
    # However, the goal is to pass messages directly.
    # For this refactor, we assume `messages` is the primary input.
    # If system_prompt and user_prompt_text were still separate, you'd do:
    # messages_list = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_text}]
    print(">>>> RUNNING REFACTORED decide_json_action V2 <<<<") # Add a unique marker

    raw_response_text, error = _make_llm_api_call(
        agent_name, messages, # Pass messages list
        llm_provider, llm_model, llm_provider_endpoint, 
        max_output_tokens, context_for_token_counting
    )
    if error:
        return {"action_type": "error", "data": {"message": error, "agent_name": agent_name}}

    potential_json_string = raw_response_text.strip()
    parsed_response = None
    try:
        if potential_json_string.startswith("```json"):
            potential_json_string = potential_json_string[7:-3].strip()
        elif potential_json_string.startswith("```"):
            potential_json_string = potential_json_string[3:-3].strip()
        
        try:
            parsed_response = json.loads(potential_json_string)
        except json.JSONDecodeError:
            # Try to extract JSON if it's embedded
            match = re.search(r"(\{[\s\S]*\})(?=\s*$|\n)", raw_response_text)
            if match:
                json_like_string = match.group(1).strip()
                parsed_response = json.loads(json_like_string)
            else:
                raise # Re-raise if no JSON found
        
        # Basic validation for expected structure
        if "action_type" not in parsed_response or "data" not in parsed_response:
            raise ValueError(f"Response from agent '{agent_name}' missing 'action_type' or 'data' keys.")
        return parsed_response
    except Exception as e: # Catch broader errors including ValueError from check above
        print(f"FINAL JSON parsing/validation error for agent '{agent_name}': {e}")
        print(f"Problematic text: '{potential_json_string}'")
        return {"action_type": "error", "data": {"message": f"JSON/Validation Error: {e}", "raw_response": raw_response_text, "agent_name": agent_name}}

def get_text_response(agent_name, messages, # Changed from system_prompt, user_prompt_text
                      llm_provider, llm_model, llm_provider_endpoint, 
                      max_output_tokens, context_for_token_counting):
    # Similar to decide_json_action, assume `messages` is the primary input.
    # If system_prompt and user_prompt_text were still separate, you'd construct messages:
    # messages_list = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt_text}]

    raw_response_text, error = _make_llm_api_call(
        agent_name, messages, # Pass messages list
        llm_provider, llm_model, llm_provider_endpoint, 
        max_output_tokens, context_for_token_counting
    )
    if error:
        return f"Error from agent '{agent_name}': {error}"
    return strip_think_tags(raw_response_text)

def _format_main_decision_prompt_text(prompt_parts):
    """Formats the complex user prompt for the MainDecision 'conceptual' agent."""
    # current_utc_datetime_str is added here
    return f"""Current UTC Date and Time for your situational awareness: {prompt_parts.get('current_utc_datetime_str')}
Consider this current date and time if the user's query implies recency or is time-sensitive (e.g., "today's news", "current stock price").

Original Question: {prompt_parts.get('user_question')}
Current Question/Task: {prompt_parts.get('current_question')}
Is current question the original question? {'Yes' if prompt_parts.get('is_user_question_context') else 'No'}

Dynamically Retrieved Context from Session Memory (most similar to current question/task):
{prompt_parts.get('dynamically_retrieved_context_str', "None available or query not applicable.")}

Available Knowledge Snippets (most recent {MAX_KNOWLEDGE_SNIPPETS_IN_PROMPT}):
{prompt_parts.get('knowledge_snippets_str', "No knowledge gathered yet.")}

URLs available to visit (max {MAX_URLS_TO_VISIT_IN_PROMPT} shown):
{prompt_parts.get('urls_to_visit_str', "None")}

Previous bad attempts for the CURRENT question/task (if any):
{prompt_parts.get('bad_attempts_for_current_q_str', "None")}

Disabled actions for next step:
Reflect disabled: {prompt_parts.get('disable_reflect_next')}
Search disabled: {prompt_parts.get('disable_search_next')}
Visit disabled: {prompt_parts.get('disable_visit_next')}"""


def _format_knowledge_snippets_for_prompt(knowledge_base):
    if not knowledge_base:
        return "No knowledge gathered yet."

    snippets_for_prompt = []
    for item in knowledge_base[-MAX_KNOWLEDGE_SNIPPETS_IN_PROMPT:]: # Show most recent
        item_type = item.get('type', 'knowledge')
        content_str = "N/A"

        if item_type == "stock_data":
            content_str = item.get('data_text', "Stock data format error.")
        elif item_type == "current_datetime":
            content_str = f"{item.get('datetime', 'N/A')} (Timezone: {item.get('timezone', 'N/A')})"
        elif item_type == "search_result_snippet":
            content_str = item.get('snippet', "No snippet text.")
        elif item_type == "web_content":
            content_str = item.get('content', "No web content text.")
        elif item_type == "intermediate_answer":
            content_str = item.get('answer', "No intermediate answer text.")
        else: # Generic fallback for other types or if 'content' is a primary key
            content_str = str(item.get('content', str(item))) # Fallback to full item string if no specific content key
        
        if len(content_str) > 200:
            truncated_content = content_str[:197] + "..."
        else:
            truncated_content = content_str
        snippets_for_prompt.append(f"- {item_type}: {truncated_content}")
    return "\n".join(snippets_for_prompt)

# --- Agent System Prompts ---
MAIN_DECISION_AGENT_SYSTEM_PROMPT = f"""
You are TinyResearch, an advanced AI research agent designed to answer complex questions by breaking them down, searching the web, visiting URLs, and synthesizing information.
You are specialized in multistep reasoning.
Your goal is to provide a definitive answer to the original question with references.
Pay close attention to all contextual information provided in the user's message, including any current date/time information, knowledge snippets, and available URLs, to make the best decision.
Based on the user-provided context (original question, current task, knowledge, URLs, bad attempts, disabled actions), decide your next action.
Your response MUST be a valid JSON object with two keys: "action_type" and "data".

Possible "action_type" values:

**IMPORTANT: First, check the 'Dynamically Retrieved Context from Session Memory'. If it directly and sufficiently answers the 'Current Question/Task', you MUST use the 'answer' action_type and formulate your answer based on that retrieved context.**

1. "answer": Provide an answer to the CURRENT question/task.
   - "data": {{"text": "Your answer...", "is_definitive": true_or_false, "has_references": true_or_false, "is_for_original": true_or_false}}
2. "reflect": Break down the CURRENT question/task into simpler sub-questions.
   - "data": {{"sub_questions": ["Sub-question 1", "Sub-question 2"]}} (Max {MAX_SUB_QUESTIONS_PER_REFLECT} sub-questions)
3. "search": Perform a web search.
   - "data": {{"query": "Your concise search query"}}
4. "visit": Visit one or more URLs.
   - "data": {{"urls": ["url_to_visit_1", "url_to_visit_2"]}} (Max {MAX_URLS_PER_VISIT_ACTION} URLs per visit action from available URLs)
5. "get_current_datetime": Get the current date and time, optionally for a specific timezone.
   - "data": {{"timezone": "IANA_timezone_name"}} (e.g., "Europe/Rome", "America/New_York". If timezone is missing or invalid, UTC will be used.)
6. "get_stock_price": Get the current stock price and related information for a given ticker symbol.
   - "data": {{"ticker": "STOCK_TICKER_SYMBOL"}} (e.g., "NVDA", "AAPL")

Choose the most appropriate action. Be strategic.

- **CRITICAL: If 'Available Knowledge Snippets' contain data (e.g., from 'stock_data' or 'current_datetime' tools) that directly answers the CURRENT question/task, you MUST use that data to form an 'answer' action. DO NOT re-run the same tool if the required data is already present in the snippets from a previous step.**
- If the CURRENT question/task is the USER's original question and you believe you have a definitive answer WITH references, set "is_definitive": true and "has_references": true for the "answer" action.
- Ground the answer with external web content.
- Read full content from URLs and get the fulltext, knowledge, clues, hints for better answer the question.  
- For queries about specific, verifiable facts that can change rapidly (e.g., stock prices, current statistics, live scores), if you have identified relevant source URLs from a search, **you SHOULD prioritize the 'visit' action** to retrieve the latest information directly from one of those source pages before attempting to 'answer'. Do not rely solely on potentially outdated search snippets for such data.
- **For stock price queries, prefer using the 'get_stock_price' action if available.**
- When providing an 'answer' for volatile data like stock prices, if you set "has_references": true, it should ideally mean you are citing information obtained from a URL you have *successfully visited* or from a dedicated tool like 'get_stock_price'.
- Before searching the web, check URLs mentioned in the USER's original question, if any. 
- Choose and visit relevant URLs below for more knowledge. higher weight suggests more relevant content
- If stuck, try to 'reflect' or 'search'.
- If using 'litellm' as the provider, your entire response MUST be ONLY the JSON object. Do not include any other text, explanations, or markdown formatting.
"""

DEEPHERMES3_SYSTEM_PROMPT = """You are a deep thinking AI, you may use extremely long chains of thought to deeply consider the problem and deliberate with yourself via systematic reasoning processes to help come to a correct solution prior to answering. You should enclose your thoughts and internal monologue inside <think> </think> tags, and then provide your solution or response to the problem."""

# Store original system prompts to revert if needed, or select dynamically
ORIGINAL_MAIN_DECISION_AGENT_SYSTEM_PROMPT = MAIN_DECISION_AGENT_SYSTEM_PROMPT

SEARCH_STRATEGY_AGENT_SYSTEM_PROMPT = f"""You are a Search Strategy Specialist.
Given the current research question and a proposed general search query, determine the most appropriate search engine and refine the query if necessary.
Your entire response MUST be ONLY the JSON object.
You MUST respond with a JSON object. This JSON object MUST have two top-level keys: "action_type" and "data".
1.  The "action_type" key's value MUST be the string "search_strategy_determined".
2.  The "data" key's value MUST be another JSON object. This inner object MUST contain two keys:
    a.  "engine_to_use": A string representing the chosen search engine. It MUST be one of: "DuckDuckGo", "Wikipedia", "Arxiv", "Pubmed", "GoogleScholar".
    b.  "refined_query": A string representing the refined search query.

Example of your EXACT output format:
{{"action_type": "search_strategy_determined", "data": {{"engine_to_use": "GoogleScholar", "refined_query": "latest breakthroughs in quantum computing 2023"}}}}
"""

ANSWER_EVALUATOR_AGENT_SYSTEM_PROMPT = f"""You are an Answer Evaluator.
Given an Original Question and a Proposed Answer, determine if the Proposed Answer is a good, definitive, and satisfactory response.
Consider completeness, correctness, and whether it directly addresses all parts of the Original Question.

Here are some guidelines:
- If the Original Question is a direct recall question (e.g., "What is my name?", "What was the previous topic I mentioned?") and the Proposed Answer directly provides the recalled information (e.g., "Your name is Piero.", "You previously mentioned X."), this IS a good and direct answer.
IMPORTANT: If the Original Question asks for the current date/time (e.g., "What time is it?"), and the Proposed Answer provides a specific date/time string (e.g., "The current time is YYYY-MM-DD HH:MM:SS TZ"), this IS a good and direct answer. Assume the provided time is accurate as if just fetched by a tool.
Similarly, if the question asks for a stock price and the answer provides it, assume it's good.

Your response MUST be a JSON object with two top-level keys: "action_type" and "data".
1.  The "action_type" key's value MUST be the string "answer_evaluation_result".
2.  The "data" key's value MUST be another JSON object containing two keys:
    a.  "is_good_answer": A boolean (true or false).
    b.  "reason": A brief string explaining your evaluation.

Example of your EXACT output format:
{{"action_type": "answer_evaluation_result", "data": {{"is_good_answer": true, "reason": "The answer is comprehensive and accurate."}}}}"""

SUMMARIZER_AGENT_SYSTEM_PROMPT = """You are a Summarizer Agent.
Based on all the knowledge gathered so far (provided by the user), provide the best possible final answer to the original question.
If possible, include references or sources for the information.
Your response should be a direct textual answer.
"""
# --- End Agent System Prompts ---


def strip_think_tags(text_content):
    stripped_content = re.sub(r"<think>[\s\S]*?</think>\s*", "", text_content, flags=re.IGNORECASE)
    return stripped_content.strip()

# --- Context Initialization ---
def initialize_context_and_variables(args):
    # For shared DB from UI
    shared_vector_db_index = getattr(args, 'shared_vector_db_index', None)
    shared_vector_db_texts = getattr(args, 'shared_vector_db_texts', None)
    shared_embedding_dim = getattr(args, 'shared_embedding_dim', None)
    print("Initializing context and variables...")
    context = {
        "token_budget": args.token_budget,
        "tokens_used": 0,
        "gaps_queue": [],
        "user_question": args.user_question,
        "current_question": None,
        "knowledge_base": [],
        "bad_attempts": [],
        "urls_to_visit": [],
        "visited_urls": set(),
        "known_questions": set(),
        "disable_reflect_next": False,
        "disable_search_next": False,
        "disable_visit_next": False,
        "simulation_step": 0,
        "max_simulation_steps": args.max_simulation_steps,
        "start_time": time.time(), # Store start time
        "time_budget_seconds": args.time_budget_minutes * 60, # Store time budget in seconds
        "is_user_question_context": False, # Added comma here
        # Add new context keys for reasoning style from args
        "selected_reasoning_style": args.reasoning_style,
        "reasoning_style_active": args.reasoning_style_active,
        "embedding_model_name": args.embedding_model_name, # Store embedding model name
        # 'embedding_dim' will be populated by _initialize_vector_db -> _get_embedding_dimension
    }
    context["gaps_queue"].append({"text": args.user_question, "is_original": True})
    context["known_questions"].add(args.user_question)

    if shared_vector_db_index is not None and shared_vector_db_texts is not None and shared_embedding_dim is not None:
        print("Using shared vector DB from UI session.")
        context['vector_db_index'] = shared_vector_db_index
        context['vector_db_texts'] = shared_vector_db_texts
        context['embedding_dim'] = shared_embedding_dim
        # Add current user question to the shared DB if it's not already there (idempotency handled by known_questions for gaps)
        # For simplicity, we assume the UI might have already added it or will.
        # Or, we can add it here with a specific source.
        add_text_to_vector_db(context, args.user_question, source_info="tinyresearcher_initial_user_question")
    else:
        print("Initializing new vector DB for this TinyResearcher run.")
        _initialize_vector_db(context) # Initialize vector DB
        add_text_to_vector_db(context, args.user_question, source_info="tinyresearcher_initial_user_question")
    print(f"Context initialized for TinyResearcher. Original question: '{args.user_question}'")
    return context

# --- Budget Check ---
def check_token_budget_exceeded(context):
    print("Checking token budget...")
    exceeded = context["tokens_used"] >= context["token_budget"]
    if exceeded:
        print(f"Token budget EXCEEDED ({context['tokens_used']}/{context['token_budget']}).")
    else:
        print(f"Token budget OK ({context['tokens_used']}/{context['token_budget']}).")
    return exceeded

def check_time_budget_exceeded(context):
    print("Checking time budget...")
    elapsed_time_seconds = time.time() - context["start_time"]
    exceeded = elapsed_time_seconds >= context["time_budget_seconds"]
    if exceeded:
        print(f"Time budget EXCEEDED ({elapsed_time_seconds:.0f}s / {context['time_budget_seconds']}s).")
    else:
        print(f"Time budget OK ({elapsed_time_seconds:.0f}s / {context['time_budget_seconds']}s).")
    return exceeded

# --- Question Handling ---
def get_current_question_from_gaps(context):
    print("Getting current question from gaps...")
    if not context["gaps_queue"]:
        print("Gaps queue is empty.")
        return None
    question_item = context["gaps_queue"].pop(0) # This is the item we might re-queue
    context["current_question"] = question_item["text"]
    context["is_user_question_context"] = question_item["is_original"]
    print(f"Current question: '{context['current_question']}' (Original: {question_item['is_original']})")
    return question_item # Return the whole item

# --- Action: Answer ---
def is_answer_for_original_question(context, answer_data): # Renamed for clarity, context not strictly needed here
    is_original = answer_data.get("is_for_original", False)
    print(f"Answer check: Is for original question? {'Yes' if is_original else 'No'}")
    return is_original

def evaluate_answer_with_agent(context, answer_text, answer_evaluator_agent):
    print(f"Evaluating answer with agent: '{answer_text[:50]}...'")
    eval_user_prompt = f"""Original Question: {context['user_question']}
Proposed Answer to Original Question: {answer_text}"""
    
    eval_response = decide_json_action(
        agent_name="AnswerEvaluatorAgent",
        # Construct messages for decide_json_action
        messages=[{"role": "system", "content": ANSWER_EVALUATOR_AGENT_SYSTEM_PROMPT}, {"role": "user", "content": eval_user_prompt}],
        llm_provider=g_llm_provider,
        llm_model=g_llm_model, # Could use a faster/cheaper model
        llm_provider_endpoint=g_llm_provider_endpoint,
        max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
        context_for_token_counting=context
    )
    if eval_response.get("action_type") == "error": # Check if the call itself had an error
        print(f"Error during LLM-based answer evaluation: {eval_response.get('data', {}).get('message')}. Falling back.")
        return False # Fallback on error

    # The evaluator agent's "data" field should contain the evaluation
    eval_result_data = eval_response.get("data", {})
    is_good_eval = eval_result_data.get("is_good_answer", False)
    reason = eval_result_data.get("reason", "No reason provided by evaluator agent.")
    print(f"Agent Evaluation: Good Answer? {is_good_eval}. Reason: {reason}")
    return is_good_eval

def is_answer_definitive_check(context, is_good_evaluation): # context not strictly needed
    print(f"Is answer definitive? {'Yes' if is_good_evaluation else 'No'}")
    return is_good_evaluation

def has_references_check(context, answer_data): # context not strictly needed
    llm_claims_refs = answer_data.get("has_references", False)
    text_has_urls = bool(re.search(r'https?://\S+', answer_data.get("text", "")))
    has_refs = llm_claims_refs or text_has_urls
    print(f"Has references? {'Yes' if has_refs else 'No'}")
    return has_refs

def store_bad_attempt_reset_context(context):
    print(f"Storing bad attempt for question: '{context['current_question']}'.")
    context["bad_attempts"].append({"question": context["current_question"], "attempt_data": "Failed to get a definitive answer or required info."})
    print("Bad attempt stored.")

def store_as_intermediate_knowledge(context, answer_data):
    print(f"Storing answer for '{context['current_question']}' as intermediate knowledge: '{answer_data['text'][:50]}...'")
    context["knowledge_base"].append({"question": context["current_question"], "answer": answer_data["text"], "type": "intermediate_answer"})
    add_text_to_vector_db(context, answer_data["text"], source_info=f"intermediate_answer_to_{context['current_question']}")
    print("Intermediate knowledge stored.")

# --- Action: Reflect ---
def process_new_sub_questions(context, reflect_data):
    sub_questions = reflect_data.get("sub_questions", [])
    print(f"Processing new sub-questions: {sub_questions}")
    if context["disable_reflect_next"]:
        print("Reflection is disabled for this step. No sub-questions processed.")
        return []
    return sub_questions

def new_unique_questions_check(context, new_questions):
    print("Deduplicating new questions...")
    unique_new_ones = []
    for q_text in new_questions:
        if q_text not in context["known_questions"]:
            unique_new_ones.append(q_text)
            context["known_questions"].add(q_text)
    if unique_new_ones:
        print(f"Found {len(unique_new_ones)} new unique questions: {unique_new_ones}")
    else:
        print("No new unique questions found.")
    return unique_new_ones

def add_to_gaps_queue(context, unique_questions_texts):
    print(f"Adding {len(unique_questions_texts)} unique questions to gaps queue.")
    for q_text in unique_questions_texts:
        context["gaps_queue"].append({"text": q_text, "is_original": False}) 

def disable_reflect_for_next_step(context):
    print("Disabling 'reflect' action for the next ModelGen step.")
    context["disable_reflect_next"] = True

# --- Action: Search ---
def execute_search(context, query): # Takes query directly now
    print(f"Executing web search (DuckDuckGo) for query: '{query}'")
    if context["disable_search_next"]: # This flag might be less relevant if search strategy agent is used
        print("Search is disabled for this step. No search performed.")
        return []
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=MAX_SEARCH_RESULTS) 
            urls = [r['href'] for r in results if 'href' in r]
            print(f"Search found URLs: {urls}")
            for res in results[:MAX_SEARCH_SNIPPETS_TO_STORE]: 
                context["knowledge_base"].append({"query": query, "title": res.get('title'), "snippet": res.get('body'), "source_url": res.get('href'), "type": "search_result_snippet"})
                add_text_to_vector_db(context, f"{res.get('title', '')} {res.get('body', '')}", source_info=f"search_snippet_for_{query}")
            return urls
    except Exception as e:
        print(f"Error during web search: {e}")
        return []

# --- Specialized Search Functions (Placeholders) ---
def execute_wikipedia_search(context, query):
    print(f"Executing Wikipedia search for query: '{query}'")
    # TODO: Implement Wikipedia search (e.g., using 'wikipedia' library)
    # try:
    #     import wikipedia
    #     page = wikipedia.page(query, auto_suggest=False) # auto_suggest=False for more direct query
    #     summary = wikipedia.summary(query, sentences=2)
    #     context["knowledge_base"].append({"query": query, "title": page.title, "snippet": summary, "source_url": page.url, "type": "wikipedia_snippet"})
    #     return [page.url]
    # except Exception as e:
    #     print(f"Error during Wikipedia search: {e}")
    #     return []
    print("Wikipedia search (mocked) returning empty list.")
    return []

def execute_arxiv_search(context, query):
    print(f"Executing Arxiv search for query: '{query}'")
    # TODO: Implement Arxiv search (e.g., using 'arxiv' library)
    print("Arxiv search (mocked) returning empty list.")
    return []
# --- End Specialized Search Functions ---

def new_urls_found_check(context, found_urls):
    new_urls = [url for url in found_urls if url not in context["visited_urls"] and url not in context["urls_to_visit"]]
    if new_urls:
        print(f"Found {len(new_urls)} new URLs: {new_urls}")
    else:
        print("No new URLs found.")
    return new_urls

def store_urls_for_future_visits(context, new_urls):
    print(f"Storing {len(new_urls)} new URLs for future visits.")
    context["urls_to_visit"].extend(new_urls)

def disable_search_for_next_step(context):
    print("Disabling 'search' action for the next ModelGen step.")
    context["disable_search_next"] = True

# --- Action: Get Current Datetime ---
def execute_get_current_datetime(context, datetime_data):
    target_timezone_str = datetime_data.get("timezone", "UTC")
    try:
        target_tz = pytz.timezone(target_timezone_str)
    except pytz.UnknownTimeZoneError:
        print(f"Warning: Unknown timezone '{target_timezone_str}'. Defaulting to UTC.")
        target_tz = pytz.utc

    now_utc = datetime.now(pytz.utc)
    now_target_tz = now_utc.astimezone(target_tz)
    
    formatted_datetime = now_target_tz.strftime("%Y-%m-%d %H:%M:%S %Z%z")
    print(f"Current datetime in {target_tz.zone}: {formatted_datetime}")

    # Add this information to the knowledge base
    knowledge_entry = {"type": "current_datetime", "timezone": target_tz.zone, "datetime": formatted_datetime, "query_timezone": target_timezone_str}
    context["knowledge_base"].append(knowledge_entry)
    return formatted_datetime # Or the knowledge_entry dict

# --- Action: Get Stock Price ---
def execute_get_stock_price(context, stock_data):
    ticker_symbol = stock_data.get("ticker")
    if not ticker_symbol:
        print("Error: Ticker symbol not provided for get_stock_price action.")
        return None

    print(f"Fetching stock data for ticker: {ticker_symbol} using yfinance...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info # Fetches a lot of info, might be slow or rate-limited
        
        # More reliable way to get current price if .info is too much or fails
        # hist = ticker.history(period="1d", interval="1m") # Get last minute data for today
        # current_price = hist['Close'].iloc[-1] if not hist.empty else info.get('currentPrice', info.get('regularMarketPreviousClose'))
        
        # Using info for simplicity, but be aware it can be extensive
        current_price = info.get('currentPrice', info.get('regularMarketPrice', info.get('regularMarketPreviousClose')))
        previous_close = info.get('regularMarketPreviousClose')
        day_change_pct = ((current_price - previous_close) / previous_close) * 100 if current_price and previous_close else None
        currency = info.get('currency', 'USD')

        result_text = f"Stock data for {ticker_symbol}: Current Price: {current_price:.2f} {currency}."
        if day_change_pct is not None:
            result_text += f" Day Change: {day_change_pct:+.2f}%."
        print(result_text)
        context["knowledge_base"].append({"type": "stock_data", "ticker": ticker_symbol, "data_text": result_text, "raw_info": info}) # Storing raw_info might be too verbose
        return result_text
    except Exception as e:
        print(f"Error fetching stock data for {ticker_symbol} with yfinance: {e}")
        return None

# --- Action: Visit ---
def visit_urls(context, visit_data):
    urls_to_process = visit_data.get("urls", [])
    if not isinstance(urls_to_process, list): 
        urls_to_process = [urls_to_process] if urls_to_process else []

    print(f"Attempting to visit URLs: {urls_to_process}")
    if context["disable_visit_next"]:
        print("Visit is disabled for this step. No URLs visited.")
        return []
    
    new_content_pieces = []
    headers = {'User-Agent': context.get("web_browser_user_agent", DEFAULT_WEB_BROWSER_USER_AGENT)} 

    for url in urls_to_process:
        if not url or not isinstance(url, str):
            print(f"Skipping invalid URL: {url}")
            continue
        if url in context["visited_urls"]:
            print(f"URL already visited: {url}")
            continue
        try:
            print(f"Visiting {url} ...")
            response = requests.get(url, headers=headers, timeout=REQUESTS_TIMEOUT)
            response.raise_for_status() 
            soup = BeautifulSoup(response.content, 'html.parser')
            for script_or_style in soup(["script", "style"]):
                script_or_style.decompose()
            text_content = soup.get_text(separator='\n', strip=True)
            current_max_url_content_length = context.get("max_url_content_length", DEFAULT_MAX_URL_CONTENT_LENGTH)
            if len(text_content) > current_max_url_content_length:
                text_content = text_content[:current_max_url_content_length] + " ... (content truncated)"
            print(f"Extracted content (first {DEFAULT_MAX_URL_CONTENT_LENGTH} chars): {text_content[:DEFAULT_MAX_URL_CONTENT_LENGTH]}...")
            new_content_pieces.append({"url": url, "content": text_content})
            context["visited_urls"].add(url)
        except requests.RequestException as e:
            print(f"Failed to visit URL {url}: {e}")
        except Exception as e:
            print(f"Error processing content from {url}: {e}")
        finally:
            if url in context["urls_to_visit"]:
                 try: context["urls_to_visit"].remove(url) 
                 except ValueError: pass
    return new_content_pieces

def new_content_found_check(context, content_pieces): # context not strictly needed
    if content_pieces: print(f"Found {len(content_pieces)} new content pieces.")
    else: print("No new content found from visits.")
    return bool(content_pieces) 

def store_content_as_knowledge(context, content_pieces):
    print(f"Storing {len(content_pieces)} new content pieces as knowledge.")
    for piece in content_pieces:
        context["knowledge_base"].append({"source_url": piece["url"], "content": piece["content"], "type": "web_content"})
        add_text_to_vector_db(context, piece["content"], source_info=f"web_content_from_{piece['url']}")

def disable_visit_for_next_step(context):
    print("Disabling 'visit' action for the next ModelGen step.")
    context["disable_visit_next"] = True

# --- Beast Mode & Final Answer ---
def enter_beast_mode(context, summarizer_agent):
    print("Entering Beast Mode due to budget exhaustion!")
    return generate_final_answer_with_agent(context, "budget exceeded", summarizer_agent)

def generate_final_answer_with_agent(context, reason, summarizer_agent):
    print(f"Generating final answer with agent ({reason})...")

    knowledge_source_for_summary = []
    source_type_for_prompt = "internal knowledge_base"

    # Prioritize Tiny Researcher's own gathered knowledge if it exists
    # and if the reason for summarizing isn't just an empty gaps queue without internal work.
    if context["knowledge_base"] and reason != "gaps queue empty, synthesizing from knowledge":
        knowledge_source_for_summary = context["knowledge_base"][-10:] # Last 10 items
    elif context.get("vector_db_texts"): # Fallback to shared session memory
        print("Attempting to use shared vector_db_texts for final summary.")
        knowledge_source_for_summary = context["vector_db_texts"][-10:] # Last 10 items from shared DB
        source_type_for_prompt = "shared session memory (vector_db_texts)"

    # Check for direct answers from structured data (datetime, stock)
    # This check should ideally use the chosen knowledge_source_for_summary
    if knowledge_source_for_summary: # Check if we have any source to look into
        question_lower = context["user_question"].lower()
        if "time" in question_lower or "date" in question_lower:
            datetime_knowledge = None
            for item in reversed(knowledge_source_for_summary): # Check the chosen source
                if item.get("type") == "current_datetime":
                    query_tz_str = item.get('query_timezone', item.get('timezone', ''))
                    city_from_tz = query_tz_str.split('/')[-1].lower().replace('_', ' ') if '/' in query_tz_str else ""
                    if (city_from_tz and city_from_tz in question_lower) or (query_tz_str.lower() in question_lower):
                        datetime_knowledge = item
                        break
            if datetime_knowledge:
                direct_answer = f"The current date and time in {datetime_knowledge.get('query_timezone', datetime_knowledge.get('timezone', 'the specified timezone'))} is {datetime_knowledge.get('datetime')}."
                print(f"Prioritizing direct datetime knowledge as final answer: {direct_answer}")
                return direct_answer

        stock_keywords = ["stock", "price", "market", "shares", "ticker"] # Add more if needed
        is_stock_query = any(keyword in question_lower for keyword in stock_keywords)
        if is_stock_query:
            stock_data_knowledge = None
            for item in reversed(knowledge_source_for_summary): # Check the chosen source
                if item.get("type") == "stock_data":
                    ticker_in_knowledge = item.get("ticker", "").lower()
                    # If question mentions the ticker, or if it's the only stock data we have
                    if (ticker_in_knowledge and ticker_in_knowledge in question_lower) or not stock_data_knowledge:
                        stock_data_knowledge = item
                        if ticker_in_knowledge and ticker_in_knowledge in question_lower: # Prefer specific match
                            break 
            if stock_data_knowledge and stock_data_knowledge.get("data_text"):
                direct_answer = stock_data_knowledge.get("data_text")
                print(f"Prioritizing direct stock_data knowledge as final answer: {direct_answer}")
                return direct_answer

    if knowledge_source_for_summary: # If we have any knowledge (internal or shared)
        # Prepare user prompt for summarizer agent
        knowledge_summary_for_prompt = "\n".join([
            # Adjust formatting based on typical keys in knowledge_base vs vector_db_texts
            f"- Source: {item.get('source', item.get('type', 'info'))}, Content: {str(item.get('text', item.get('snippet', item.get('content', item.get('answer', 'N/A')))))[:150]}..."
            for item in knowledge_source_for_summary # Use the selected source
        ])
        
        summary_user_prompt = f"""Original Question: {context['user_question']}
Reason for this summary: {reason}.

You are tasked with providing the best possible final answer to the 'Original Question' based *only* on the 'Available Information Snippets' provided below. These snippets are from {source_type_for_prompt}.

**If the 'Original Question' is a direct recall question (e.g., "What is my name?", "What was the previous topic?") and a snippet directly answers it, provide that answer directly.**
Otherwise, synthesize a comprehensive answer.

If possible, include references or sources for the information.
The most crucial pieces of information are likely the most recently gathered ones.

Available Information Snippets (most recent up to 10 from {source_type_for_prompt}):
{knowledge_summary_for_prompt}
"""
        # The line below was causing the AttributeError and is now removed.
        # final_answer_text = summarizer_agent.get_text_response(summary_user_prompt, context) 
        final_answer_text = get_text_response(
            agent_name=summarizer_agent,
            messages=[{"role": "system", "content": SUMMARIZER_AGENT_SYSTEM_PROMPT}, {"role": "user", "content": summary_user_prompt}],
            llm_provider=g_llm_provider,
            llm_model=g_llm_model,
            llm_provider_endpoint=g_llm_provider_endpoint,
            max_output_tokens=LLM_SUMMARIZER_MAX_OUTPUT_TOKENS,
            context_for_token_counting=context
        )
        # The raw response is already printed by _make_llm_api_call, get_text_response handles stripping
        return final_answer_text
    else:
        final_answer_text = f"Final answer for '{context['user_question']}'. Reason: {reason}. No significant knowledge gathered (neither in internal knowledge_base nor shared session memory) to synthesize an answer."
        print(f"\nFinal Answer: {final_answer_text}")
        return final_answer_text

# --- Main Flow Logic ---
def run_deep_search_agent(args):
    global g_llm_provider, g_llm_model, g_llm_provider_endpoint, g_gemini_api_key

    print(f"--- Starting SmolDeepSearch CLI with {args.llm_provider} ({args.llm_model}) ---")
    
    # Initialize global LLM settings from args
    g_llm_provider = args.llm_provider
    g_llm_model = args.llm_model
    g_llm_provider_endpoint = args.llm_provider_endpoint
    if args.llm_provider == "gemini":
        g_gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not g_gemini_api_key:
            print("ERROR: GEMINI_API_KEY environment variable is not set for 'gemini' provider.")
            exit(1)
        genai.configure(api_key=g_gemini_api_key)

    # Get current UTC date and time once for the entire run
    current_utc_datetime_str = datetime.now(pytz.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    context = initialize_context_and_variables(args)
    context["web_browser_user_agent"] = args.web_browser_user_agent
    context["max_url_content_length"] = args.max_url_content_length
    # No agent objects to initialize anymore.
    # System prompts are constants. LLM configs are from args (via globals).
    # Agent names are passed as strings to the LLM call functions.
    # Max output tokens are passed as arguments (LLM_MAX_OUTPUT_TOKENS or LLM_SUMMARIZER_MAX_OUTPUT_TOKENS).

    final_result = None
    # Initialize a flag for successful handling before the action dispatch, reset each loop
    current_question_successfully_handled_this_step = False

    while context["simulation_step"] < args.max_simulation_steps:
        print(f"\n--- Loop Iteration: {context['simulation_step'] + 1} ---")
        print(f"Tokens used so far: {context['tokens_used']}/{context['token_budget']}")

        if check_token_budget_exceeded(context) or check_time_budget_exceeded(context):
            final_result = enter_beast_mode(context, "SummarizerAgent") # Pass agent name for logging
            break 

        # Reset for the current iteration
        current_question_successfully_handled_this_step = False

        # Get the current question item to process
        current_question_item = get_current_question_from_gaps(context)
        if not current_question_item:
            print("No more questions in the GAPS queue.")
            if context["knowledge_base"]:
                 final_result = generate_final_answer_with_agent(context, "gaps queue empty, synthesizing from knowledge", "SummarizerAgent")
            else:
                 final_result = "No questions left and no knowledge gathered to form an answer."
            break 
        # context["current_question"] and context["is_user_question_context"] are set by get_current_question_from_gaps
        
        # Prepare user prompt parts for the main decision agent
        retrieved_context_from_vector_db_str = query_vector_db(context, context["current_question"]) # Query Vector DB
        # Ensure this print goes to the UI log via redirect_stdout
        print(f"DEBUG_TR_CONTEXT: Retrieved for '{context['current_question']}':\n---\n{retrieved_context_from_vector_db_str}\n---")
        knowledge_snippets_str = _format_knowledge_snippets_for_prompt(context["knowledge_base"])
        urls_to_visit_str = ", ".join(context["urls_to_visit"][:MAX_URLS_TO_VISIT_IN_PROMPT]) if context["urls_to_visit"] else "None"
        bad_attempts_for_current_q_list = [f"- Attempt: {attempt['attempt_data']}" for attempt in context.get("bad_attempts", []) if attempt.get("question") == context["current_question"]]
        bad_attempts_for_current_q_str = "\n".join(bad_attempts_for_current_q_list) if bad_attempts_for_current_q_list else "None"

        agent_input_context = {
            "current_utc_datetime_str": current_utc_datetime_str, # Added here
            "user_question": context["user_question"],
            "current_question": context["current_question"],
            "is_user_question_context": context["is_user_question_context"],
            "dynamically_retrieved_context_str": retrieved_context_from_vector_db_str, # Add retrieved context
            "knowledge_snippets_str": knowledge_snippets_str,
            "urls_to_visit_str": urls_to_visit_str,
            "bad_attempts_for_current_q_str": bad_attempts_for_current_q_str,
            "disable_reflect_next": context["disable_reflect_next"],
            "disable_search_next": context["disable_search_next"],
            "disable_visit_next": context["disable_visit_next"],
        }
        
        main_decision_user_prompt_text = _format_main_decision_prompt_text(agent_input_context)
        
        # Determine system prompt based on reasoning style
        current_main_system_prompt = ORIGINAL_MAIN_DECISION_AGENT_SYSTEM_PROMPT
        if context.get("selected_reasoning_style") == "DeepHermes3" and context.get("reasoning_style_active"):
            current_main_system_prompt = DEEPHERMES3_SYSTEM_PROMPT
            print("Using DeepHermes3 system prompt for MainDecisionAgent.")

        model_response = decide_json_action(
            agent_name="MainDecisionAgent",
            # Construct messages list for the MainDecisionAgent
            messages=[{"role": "system", "content": current_main_system_prompt}, {"role": "user", "content": main_decision_user_prompt_text}],
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            llm_provider_endpoint=args.llm_provider_endpoint,
            max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
            context_for_token_counting=context
        )

        action_type = model_response.get("action_type")
        action_data = model_response.get("data")

        if action_type == "error":
            print(f"Error from Agent ({model_response.get('data',{}).get('agent_name', 'Unknown')}): {action_data.get('message') if action_data else 'Unknown error'}. Skipping.")
            store_bad_attempt_reset_context(context) # Store as bad attempt for current question
            context["simulation_step"] += 1
            time.sleep(LOOP_PAUSE_SECONDS / 2)
            continue
        
        print(f"MainDecisionAgent chose action: '{action_type}' with data: {str(action_data)[:100]}...")

        if action_type == "answer":
            if is_answer_for_original_question(context, action_data):
                evaluation_is_good = evaluate_answer_with_agent(context, action_data.get("text", ""), "AnswerEvaluatorAgent")
                if evaluation_is_good: 
                    current_question_successfully_handled_this_step = True # Mark as handled
                    if action_data.get("is_definitive"): # And if the MainDecisionAgent also marked it as definitive
                        final_result = action_data["text"] 
                        print(f"Definitive answer found for original question (Main Agent: Definitive, Evaluator: Good).")
                        # Break will be handled by the check after re-queue logic
                    else:
                        #print("Answer is definitive for original question but lacks references.")
                        # Good answer for original question, but MainDecisionAgent didn't mark it definitive.
                        print("Good answer for original question, but not marked definitive by Main Agent. Storing as intermediate.")
                        store_as_intermediate_knowledge(context, action_data) # Treat as progress
                else: # evaluation_is_good is False
                    print("Answer for original question evaluated as not good.")
                    store_bad_attempt_reset_context(context)
                    # current_question_successfully_handled_this_step remains False
            else: # Not for original question (i.e., for a sub-question)
                print("Answer is for a sub-question. Storing as intermediate knowledge.")
                store_as_intermediate_knowledge(context, action_data)
                current_question_successfully_handled_this_step = True # Mark as handled
       
        elif action_type == "reflect":
            if context["disable_reflect_next"]:
                print("Reflect chosen, but it's disabled. Storing as bad attempt.")
                store_bad_attempt_reset_context(context)
                # current_question_successfully_handled_this_step remains False
            else:
                sub_questions = process_new_sub_questions(context, action_data)
                unique_new_q = new_unique_questions_check(context, sub_questions)
                if unique_new_q: add_to_gaps_queue(context, unique_new_q)
                else: disable_reflect_for_next_step(context)
                current_question_successfully_handled_this_step = True # Mark as handled
        
        elif action_type == "search":
            if context["disable_search_next"]:
                print("Search chosen, but it's disabled. Storing as bad attempt.")
                store_bad_attempt_reset_context(context)
            else:
                # --- Agentic Call to Choose Specialized Search Engine ---
                search_strategy_user_prompt = f"""Current Research Question: "{context['current_question']}"
Proposed General Search Query: "{action_data.get('query', '')}\""""
                
                print("Determining search strategy with SearchStrategyAgent...")
                strategy_agent_response = decide_json_action(
                    agent_name="SearchStrategyAgent",
                    # Construct messages for decide_json_action
                    messages=[{"role": "system", "content": SEARCH_STRATEGY_AGENT_SYSTEM_PROMPT}, {"role": "user", "content": search_strategy_user_prompt}],
                    llm_provider=args.llm_provider,
                    llm_model=args.llm_model, # Could use a faster/cheaper model
                    llm_provider_endpoint=args.llm_provider_endpoint,
                    max_output_tokens=LLM_MAX_OUTPUT_TOKENS,
                    context_for_token_counting=context
                )


                chosen_engine = "DuckDuckGo" # Default
                refined_query = action_data.get('query', '') # Default

                if strategy_agent_response.get("action_type") != "error":
                    strategy_data = strategy_agent_response.get("data", {})
                    chosen_engine = strategy_data.get("engine_to_use", "DuckDuckGo").lower()
                    refined_query = strategy_data.get("refined_query", refined_query)
                else:
                    print(f"Error from SearchStrategyAgent: {strategy_agent_response.get('data',{}).get('message')}. Defaulting to DuckDuckGo.")

                print(f"Search Strategy: Engine='{chosen_engine}', Query='{refined_query}'")
                found_urls = []
                if chosen_engine == "wikipedia":
                    found_urls = execute_wikipedia_search(context, refined_query)
                elif chosen_engine == "arxiv":
                    found_urls = execute_arxiv_search(context, refined_query)
                # TODO: Add Pubmed, GoogleScholar etc.
                else: # Default to general web search (DuckDuckGo)
                    found_urls = execute_search(context, refined_query) # Pass refined_query
                
                new_urls = new_urls_found_check(context, found_urls)
                if new_urls: store_urls_for_future_visits(context, new_urls)
                else: disable_search_for_next_step(context)

        elif action_type == "get_current_datetime":
            # No specific disable flag for this yet, but could be added
            execute_get_current_datetime(context, action_data)

        elif action_type == "get_stock_price":
            # No specific disable flag for this yet
            execute_get_stock_price(context, action_data)

        elif action_type == "visit":
            if context["disable_visit_next"]:
                print("Visit chosen, but it's disabled. Storing as bad attempt.")
                store_bad_attempt_reset_context(context)
            elif not context["urls_to_visit"] and not action_data.get("urls"):
                print("Visit chosen, but no URLs available. Disabling visit for next.")
                disable_visit_for_next_step(context)
            else:
                urls_from_llm = action_data.get("urls", [])
                if not isinstance(urls_from_llm, list): urls_from_llm = [urls_from_llm] if urls_from_llm else []
                
                valid_urls_to_visit_action = [u for u in urls_from_llm if u in context["urls_to_visit"]]
                
                if not valid_urls_to_visit_action and context["urls_to_visit"]:
                    print("LLM did not provide valid URLs for visit from context, using first available.")
                    valid_urls_to_visit_action = context["urls_to_visit"][:1]
                
                if valid_urls_to_visit_action:
                    action_data["urls"] = valid_urls_to_visit_action
                    content_pieces = visit_urls(context, action_data)
                    if new_content_found_check(context, content_pieces):
                        store_content_as_knowledge(context, content_pieces)
                    else:
                        disable_visit_for_next_step(context)
                else:
                    print("No valid URLs to visit. Disabling visit.")
                    disable_visit_for_next_step(context)
        
        else:
            print(f"Unknown or unhandled action type: {action_type}. Storing as bad attempt.")
            store_bad_attempt_reset_context(context)

        # --- Re-queue current question if not successfully handled ---
        if final_result: # If a definitive answer for the original question was found and accepted
            break

        # Actions like search, visit, get_datetime, get_stock_price, or agent errors,
        # or an answer for the original question that was evaluated as "not good"
        # mean the question itself wasn't "successfully handled" by providing a satisfactory answer or breaking it down.
        if not current_question_successfully_handled_this_step:
            print(f"Re-queuing current question '{current_question_item['text']}' as it was not successfully handled or progressed this step.")
            context["gaps_queue"].insert(0, current_question_item)
        else:
            print(f"Current question '{current_question_item['text']}' considered successfully handled or progressed for this step. Not re-queuing immediately.")

        context["simulation_step"] += 1
        context["disable_reflect_next"] = False
        context["disable_search_next"] = False
        context["disable_visit_next"] = False
        time.sleep(LOOP_PAUSE_SECONDS)

    if not final_result and context["simulation_step"] >= args.max_simulation_steps:
        print(f"\n--- Max simulation steps ({args.max_simulation_steps}) reached ---")
        final_result = generate_final_answer_with_agent(context, f"max steps ({args.max_simulation_steps}) reached", "SummarizerAgent")

    if final_result: # Add the final answer to the vector DB
        add_text_to_vector_db(context, final_result, source_info="final_answer")
        
    # Determine steps_taken for display
    # The simulation_step is 0-indexed and incremented at the end of a full loop.
    # If we break mid-loop due to finding a final_result, we want to count that partial loop as a step.
    steps_taken_for_display = context['simulation_step']
    if final_result and context['simulation_step'] < args.max_simulation_steps:
        # If a solution was found and we didn't simply run out of max_steps
        steps_taken_for_display = context['simulation_step'] + 1

    # If final_result is already set (e.g. from inside the loop), use it.
    # Otherwise, if we hit max steps and generate_final_answer_with_agent didn't produce one, set a default.
    if not final_result and context["simulation_step"] >= args.max_simulation_steps :
        final_result = "Agent finished due to max steps, but no definitive answer was synthesized."
    
    total_time_taken_seconds = time.time() - context["start_time"]

    print("\n--- SmolDeepSearch CLI Ended ---")
    output_data = {
        "user_question": args.user_question,
        "final_answer": final_result if final_result else "No definitive answer produced.",
        "tokens_used": context['tokens_used'],
        "token_budget": args.token_budget,
        "steps_taken": steps_taken_for_display,
        "max_steps": args.max_simulation_steps,
        "knowledge_base_items": len(context['knowledge_base']),
        "bad_attempts": len(context['bad_attempts']),
        "urls_visited": len(context['visited_urls']),
        "time_taken_seconds": total_time_taken_seconds,
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model
    }
    # Ensure final_result is returned from the function

    if args.output_format == "json":
        print(json.dumps(output_data, indent=2))
    elif args.output_format == "markdown":
        md_output = f"# SmolDeepSearch Report\n\n"
        md_output += f"**Original Question:** {output_data['user_question']}\n\n"
        md_output += f"## Final Answer\n{output_data['final_answer']}\n\n"
        md_output += f"## Run Details\n"
        md_output += f"- LLM Provider: {output_data['llm_provider']}\n"
        md_output += f"- LLM Model: {output_data['llm_model']}\n"
        md_output += f"- Tokens Used: {output_data['tokens_used']} / {output_data['token_budget']}\n"
        md_output += f"- Steps Taken: {output_data['steps_taken']} / {output_data['max_steps']}\n"
        md_output += f"- Knowledge Items Gathered: {output_data['knowledge_base_items']}\n"
        md_output += f"- Time Taken: {output_data['time_taken_seconds']:.2f} seconds\n"
        md_output += f"- URLs Visited: {output_data['urls_visited']}\n"
        md_output += f"- Bad Attempts: {output_data['bad_attempts']}\n"
        print(md_output)
    else: # Default to text
        print(f"\n======= FINAL OUTPUT (TEXT) =======")
        print(f"\nOriginal Question: {output_data['user_question']}")
        print(f"\nFinal Answer:\n{output_data['final_answer']}")
        print(f"\n--- Stats ---")
        print(f"LLM Provider: {output_data['llm_provider']} ({output_data['llm_model']})")
        print(f"Tokens Used: {output_data['tokens_used']} / {output_data['token_budget']}")
        print(f"Steps: {output_data['steps_taken']} / {output_data['max_steps']}")
        print(f"Knowledge Items: {output_data['knowledge_base_items']}")
        print(f"Time Taken: {output_data['time_taken_seconds']:.2f} seconds")
        print(f"URLs Visited: {output_data['urls_visited']}")
        print(f"Bad Attempts: {output_data['bad_attempts']}")
        print(f"===================================")

    return final_result # Explicitly return the final_result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TinyResearch: An AI agent for answering complex questions by breaking them down, searching, and synthesizing information.")
    parser.add_argument("user_question", metavar="user_question", type=str, help="The initial complex question to be researched by the agent.")
    parser.add_argument("--token_budget", type=int, default=DEFAULT_TOKEN_BUDGET, help=f"Max tokens to use. (default: {DEFAULT_TOKEN_BUDGET})")
    parser.add_argument("--max_simulation_steps", type=int, default=DEFAULT_MAX_SIMULATION_STEPS, help=f"Max iterations for the agent. (default: {DEFAULT_MAX_SIMULATION_STEPS})")
    parser.add_argument("--time_budget_minutes", type=int, default=DEFAULT_TIME_BUDGET_MINUTES, help=f"Max time in minutes for the agent to run. (default: {DEFAULT_TIME_BUDGET_MINUTES})")
    parser.add_argument("--web_browser_user_agent", type=str, default=DEFAULT_WEB_BROWSER_USER_AGENT, help=f"User agent for web requests. (default: '{DEFAULT_WEB_BROWSER_USER_AGENT}')")
    parser.add_argument("--max_url_content_length", type=int, default=DEFAULT_MAX_URL_CONTENT_LENGTH, help=f"Max characters to extract from a URL. (default: {DEFAULT_MAX_URL_CONTENT_LENGTH})")
    parser.add_argument("--llm_provider", type=str, choices=["gemini", "litellm"], default=DEFAULT_LLM_PROVIDER, help=f"LLM provider to use. (default: {DEFAULT_LLM_PROVIDER}, see https://docs.litellm.ai/docs/providers)")
    parser.add_argument("--llm_provider_endpoint", type=str, default=DEFAULT_LLM_PROVIDER_ENDPOINT, help=f"API endpoint for LLM provider (e.g., local Ollama). (default: {DEFAULT_LLM_PROVIDER_ENDPOINT})")
    parser.add_argument("--llm_model", type=str, default=DEFAULT_LLM_MODEL, help=f"Specific LLM model name. (default: {DEFAULT_LLM_MODEL})")
    parser.add_argument("--embedding_model_name", type=str, default=DEFAULT_LITELLM_EMBEDDING_MODEL, help=f"Embedding model name for LiteLLM (e.g., ollama/nomic-embed-text, gemini/embedding-001). (default: {DEFAULT_LITELLM_EMBEDDING_MODEL})")
    parser.add_argument("--output_format", type=str, choices=["text", "json", "markdown"], default=DEFAULT_OUTPUT_FORMAT, help=f"Output format. (default: {DEFAULT_OUTPUT_FORMAT})")
    parser.add_argument("--reasoning_style", type=str, default="LLM Default", help="Selected reasoning style (e.g., Qwen3, DeepHermes3). Used by UI.")
    parser.add_argument("--reasoning_style_active", type=bool, default=False, help="Whether the selected reasoning style is active. Used by UI.")
    
    args = parser.parse_args()
    
    run_deep_search_agent(args)
