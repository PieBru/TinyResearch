import unittest
from unittest.mock import patch, MagicMock, mock_open
import sys
import os
import json
import time
import argparse
import collections 
import numpy as np
import faiss
import requests 
from litellm.utils import ModelResponse, Choices, Message, Usage 

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tinyresearch

# Test suite version, aligned with tinyresearch.py versioning scheme
# Current target: 2024.05.10.c

# --- Mock Constants ---
MOCK_EMBEDDING_DIM = 768 
MOCK_OLLAMA_EMBEDDING_MODEL_NAME = "ollama/mock-embedding-model" 
MOCK_LITELLM_EMBEDDING_MODEL_NAME = "custom_provider/mock-litellm-embedding-model"

# --- Mock Response Creators ---
def create_mock_ollama_embedding_response(embedding_vector):
    """Creates a mock response for ollama.Client().embeddings."""
    return {"embedding": embedding_vector}

def create_mock_litellm_embedding_response(embeddings_list, model_name="mock_litellm_model", prompt_tokens=10, total_tokens=10):
    """Creates a mock response for litellm.embedding."""
    data_items = []
    for emb_vector in embeddings_list:
        item = MagicMock()
        item.embedding = emb_vector
        data_items.append(item)
    
    response = ModelResponse(model=model_name, usage=Usage(prompt_tokens=prompt_tokens, total_tokens=total_tokens))
    response.data = data_items 
    response.choices = [Choices(message=Message(content=None), finish_reason="stop")] # Ensure choices for compatibility
    return response

# --- Test Classes ---

class TestTinyResearchHelpers(unittest.TestCase):
    def test_strip_think_tags(self):
        self.assertEqual(tinyresearch.strip_think_tags("Hello <think>Hmm</think> World"), "Hello World")
        self.assertEqual(tinyresearch.strip_think_tags("No tags here"), "No tags here")
        self.assertEqual(tinyresearch.strip_think_tags("<think>Leading</think> Text"), "Text")
        self.assertEqual(tinyresearch.strip_think_tags("Text <think>Trailing</think>"), "Text")

    def test_format_knowledge_snippets_for_prompt(self):
        self.assertEqual(tinyresearch._format_knowledge_snippets_for_prompt([]), "No knowledge gathered yet.")
        kb_stock = [{"type": "stock_data", "data_text": "NVDA: $100"}]
        self.assertIn("- stock_data: NVDA: $100", tinyresearch._format_knowledge_snippets_for_prompt(kb_stock))

    def test_process_new_sub_questions(self):
        context_reflect_enabled = {"disable_reflect_next": False}
        reflect_data = {"sub_questions": ["Q1?", "Q2?"]}
        sub_q_enabled = tinyresearch.process_new_sub_questions(context_reflect_enabled, reflect_data)
        self.assertEqual(sub_q_enabled, ["Q1?", "Q2?"])

    def test_new_unique_questions_check(self):
        context = {"known_questions": set(["Existing Q1?"])}
        new_questions_all_new = ["New Q1?", "New Q2?"]
        unique = tinyresearch.new_unique_questions_check(context, new_questions_all_new)
        self.assertCountEqual(unique, ["New Q1?", "New Q2?"])

    def test_add_to_gaps_queue(self):
        context = {"gaps_queue": collections.deque()}
        questions_to_add = ["Q1", "Q2"]
        tinyresearch.add_to_gaps_queue(context, questions_to_add)
        self.assertEqual(len(context["gaps_queue"]), 2)
        self.assertEqual(context["gaps_queue"][0]["text"], "Q1")

class TestTinyResearchTools(unittest.TestCase):
    def setUp(self):
        self.context = {
            "knowledge_base": [], "tokens_used": 0, "urls_to_visit": collections.deque(), 
            "visited_urls": set(), "processed_files": set(),
            "disable_search_next": False, "disable_visit_next": False, "disable_reflect_next": False,
            "embedding_model_name": MOCK_OLLAMA_EMBEDDING_MODEL_NAME, # Default to Ollama path for these tool tests
            "gaps_queue": collections.deque()
        }
        
        self.mock_ollama_client_patcher = patch('tinyresearch.ollama.Client')
        self.mock_ollama_client_class = self.mock_ollama_client_patcher.start()
        self.mock_ollama_client_instance = self.mock_ollama_client_class.return_value
        # Default mock for ollama embeddings
        self.mock_ollama_client_instance.embeddings.return_value = create_mock_ollama_embedding_response([0.1] * MOCK_EMBEDDING_DIM)

        self.mock_litellm_embedding_patcher = patch('tinyresearch.litellm.embedding')
        self.mock_litellm_embedding = self.mock_litellm_embedding_patcher.start()
        # Default mock for litellm embeddings (fallback if a test changes context to use it)
        self.mock_litellm_embedding.return_value = create_mock_litellm_embedding_response([[0.1] * MOCK_EMBEDDING_DIM])
        
        self.mock_yfinance_ticker_patcher = patch('tinyresearch.yf.Ticker')
        self.mock_yfinance_ticker = self.mock_yfinance_ticker_patcher.start()

        # Mocking DDGS for context manager - Patching in tinyresearch's namespace
        self.mock_ddgs_patcher = patch('tinyresearch.DDGS') 
        mock_ddgs_class_constructor = self.mock_ddgs_patcher.start() # This is the mock for the DDGS class

        # This is the mock for the instance that DDGS() will produce
        self.mock_ddgs_instance_obj = MagicMock() 
        # This is the mock for the object that `__enter__` will return (the context manager object)
        self.mock_ddgs_cm_obj = MagicMock() 
        self.mock_ddgs_instance_obj.__enter__.return_value = self.mock_ddgs_cm_obj
        mock_ddgs_class_constructor.return_value = self.mock_ddgs_instance_obj
        # self.mock_ddgs_cm_obj is what `ddgs` will be in `with DDGS() as ddgs:`. Configure its `text` method in tests.
        
        self.mock_requests_get_patcher = patch('tinyresearch.requests.get')
        self.mock_requests_get = self.mock_requests_get_patcher.start()

        self.mock_datetime_patcher = patch('tinyresearch.datetime')
        self.mock_datetime_module = self.mock_datetime_patcher.start()

        self.mock_pytz_patcher = patch('tinyresearch.pytz')
        self.mock_pytz_module = self.mock_pytz_patcher.start()

        self.mock_os_path_patcher = patch('os.path') # For file operations if any tool uses it
        self.mock_os_path = self.mock_os_path_patcher.start()

        self.mock_open_patcher = patch('builtins.open', new_callable=mock_open) # For file operations
        self.mock_open = self.mock_open_patcher.start()

    def tearDown(self):
        self.mock_ollama_client_patcher.stop()
        self.mock_litellm_embedding_patcher.stop()
        self.mock_yfinance_ticker_patcher.stop()
        self.mock_ddgs_patcher.stop()
        self.mock_requests_get_patcher.stop()
        self.mock_datetime_patcher.stop()
        self.mock_pytz_patcher.stop()
        self.mock_os_path_patcher.stop()
        self.mock_open_patcher.stop()

    def test_execute_get_current_datetime(self):
        mock_now_utc = MagicMock()
        mock_now_utc.astimezone.return_value.strftime.return_value = "2023-01-01 10:00:00 UTC+0000"
        self.mock_datetime_module.now.return_value = mock_now_utc
        mock_utc_tz = MagicMock(); mock_utc_tz.zone = "UTC"
        self.mock_pytz_module.utc = mock_utc_tz
        self.mock_pytz_module.timezone.return_value = mock_utc_tz

        result = tinyresearch.execute_get_current_datetime(self.context, {"timezone": "UTC"})
        self.assertEqual(result, "2023-01-01 10:00:00 UTC+0000")
        self.assertEqual(self.context["knowledge_base"][0]["type"], "current_datetime")

    def test_execute_get_stock_price(self):
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {"currentPrice": 150.75, "regularMarketPreviousClose": 148.00, "currency": "USD", "symbol": "TEST"}
        self.mock_yfinance_ticker.return_value = mock_ticker_instance
        result = tinyresearch.execute_get_stock_price(self.context, {"ticker": "TEST"})
        self.assertIn("Stock data for TEST", result)

    def test_execute_search(self):
        # Configure the mock DDGS instance that __enter__ returns
        mock_search_results = [{'href': 'http://example.com/1', 'body': 'Snippet 1', 'title': 'Title 1'}]
        # self.mock_ddgs_cm_obj is the object that has the .text() method
        self.mock_ddgs_cm_obj.text.return_value = mock_search_results
        
        # Since context uses MOCK_OLLAMA_EMBEDDING_MODEL_NAME, mock ollama.Client().embeddings
        mock_dim_emb_ollama = create_mock_ollama_embedding_response([0.1] * MOCK_EMBEDDING_DIM)
        # Provide enough mock embeddings for dimension check + up to MAX_SEARCH_RESULTS snippets
        mock_snippet_embeddings_ollama = [
            create_mock_ollama_embedding_response([0.2 + i*0.01] * MOCK_EMBEDDING_DIM) 
            for i in range(tinyresearch.MAX_SEARCH_RESULTS)
        ]
        self.mock_ollama_client_instance.embeddings.side_effect = [
            mock_dim_emb_ollama,
            *mock_snippet_embeddings_ollama # Unpack the list of snippet embeddings
        ]
    
        returned_urls = tinyresearch.execute_search(self.context, "test query")
        self.assertIn('http://example.com/1', returned_urls)
        self.assertEqual(self.context["vector_db_index"].ntotal, 1)
        self.mock_ollama_client_instance.embeddings.assert_any_call(model="mock-embedding-model", prompt="Title 1 Snippet 1")
        self.mock_litellm_embedding.assert_not_called()

    def test_visit_urls(self):
        mock_response = MagicMock(spec=requests.Response)
        mock_response.status_code = 200
        mock_response.content = b"<html><body><p>Test content</p></body></html>"
        mock_response.raise_for_status.return_value = None
        self.mock_requests_get.return_value = mock_response
        
        mock_soup = MagicMock()
        mock_soup.get_text.return_value = "Test content"
        
        with patch('tinyresearch.BeautifulSoup', return_value=mock_soup) as mock_bs:
            content_pieces = tinyresearch.visit_urls(self.context, {"urls": ["http://example.com"]})
            self.assertEqual(len(content_pieces), 1)
            self.assertEqual(content_pieces[0]["content"], "Test content")
            mock_bs.assert_called_once()

class TestTinyResearchContextAndFlow(unittest.TestCase):
    def setUp(self):
        self.mock_ollama_client_patcher = patch('tinyresearch.ollama.Client')
        self.mock_litellm_embedding_patcher = patch('tinyresearch.litellm.embedding') 
        self.mock_litellm_completion_patcher = patch('litellm.completion')
        
        self.mock_ollama_client_class = self.mock_ollama_client_patcher.start()
        self.mock_ollama_client_instance = self.mock_ollama_client_class.return_value
        self.mock_litellm_embedding = self.mock_litellm_embedding_patcher.start()
        self.mock_litellm_completion = self.mock_litellm_completion_patcher.start()

        # Default mocks for embedding calls
        self.mock_ollama_client_instance.embeddings.return_value = create_mock_ollama_embedding_response([0.1] * MOCK_EMBEDDING_DIM)
        self.mock_litellm_embedding.return_value = create_mock_litellm_embedding_response([[0.1] * MOCK_EMBEDDING_DIM])

        self.mock_ddgs_patcher = patch('duckduckgo_search.DDGS') # Corrected
        self.mock_ddgs_class = self.mock_ddgs_patcher.start()
        self.mock_ddgs_instance = self.mock_ddgs_class.return_value.__enter__.return_value

        self.mock_open_patcher = patch('builtins.open', new_callable=mock_open)
        self.mock_open = self.mock_open_patcher.start()
        self.mock_os_path_patcher = patch('os.path')
        self.mock_os_path = self.mock_os_path_patcher.start()
        self.mock_os_path.exists.return_value = False # Default for file tests

    def tearDown(self):
        self.mock_ollama_client_patcher.stop()
        self.mock_litellm_embedding_patcher.stop()
        self.mock_litellm_completion_patcher.stop()
        self.mock_ddgs_patcher.stop()
        self.mock_open_patcher.stop()
        self.mock_os_path_patcher.stop()

    def test_initialize_context_and_variables_ollama_path(self):
        # Test with Ollama embedding model
        mock_dim_emb_ollama = create_mock_ollama_embedding_response([0.01] * MOCK_EMBEDDING_DIM)
        mock_q_emb_ollama = create_mock_ollama_embedding_response([0.02] * MOCK_EMBEDDING_DIM)
        self.mock_ollama_client_instance.embeddings.side_effect = [mock_dim_emb_ollama, mock_q_emb_ollama]
        
        mock_args = argparse.Namespace(
            token_budget=10000, user_question="Test Question Ollama?", max_simulation_steps=5,
            llm_provider="litellm", llm_model="ollama/mock-llm", llm_provider_endpoint="http://localhost:11434",
            web_browser_user_agent="TestUserAgent", max_url_content_length=1000,
            time_budget_minutes=10, reasoning_style="LLM Default", reasoning_style_active=False,
            embedding_model_name=MOCK_OLLAMA_EMBEDDING_MODEL_NAME # Explicitly use Ollama path
        )
        context = tinyresearch.initialize_context_and_variables(mock_args)
        self.assertEqual(context["user_question"], "Test Question Ollama?")
        self.assertEqual(context['embedding_dim'], MOCK_EMBEDDING_DIM)
        self.assertEqual(self.mock_ollama_client_instance.embeddings.call_count, 2)
        self.mock_litellm_embedding.assert_not_called()

    def test_initialize_context_and_variables_litellm_path(self):
        # Test with LiteLLM embedding model
        mock_dim_emb_litellm = create_mock_litellm_embedding_response([[0.01] * MOCK_EMBEDDING_DIM])
        mock_q_emb_litellm = create_mock_litellm_embedding_response([[0.02] * MOCK_EMBEDDING_DIM])
        self.mock_litellm_embedding.side_effect = [mock_dim_emb_litellm, mock_q_emb_litellm]

        mock_args = argparse.Namespace(
            token_budget=10000, user_question="Test Question LiteLLM?", max_simulation_steps=5,
            llm_provider="litellm", llm_model="ollama/mock-llm", llm_provider_endpoint="http://localhost:11434",
            web_browser_user_agent="TestUserAgent", max_url_content_length=1000,
            time_budget_minutes=10, reasoning_style="LLM Default", reasoning_style_active=False,
            embedding_model_name=MOCK_LITELLM_EMBEDDING_MODEL_NAME # Explicitly use LiteLLM path
        )
        context = tinyresearch.initialize_context_and_variables(mock_args)
        self.assertEqual(context["user_question"], "Test Question LiteLLM?")
        self.assertEqual(context['embedding_dim'], MOCK_EMBEDDING_DIM)
        self.assertEqual(self.mock_litellm_embedding.call_count, 2)
        self.mock_ollama_client_instance.embeddings.assert_not_called()


    def test_add_and_query_vector_db_ollama_model(self):
        context = {
            "tokens_used": 0, "embedding_model_name": MOCK_OLLAMA_EMBEDDING_MODEL_NAME,
            "gaps_queue": collections.deque(), "knowledge_base": [], "vector_db_texts": []
        }
        mock_dim_emb = create_mock_ollama_embedding_response([0.1] * MOCK_EMBEDDING_DIM)
        mock_text1_emb = create_mock_ollama_embedding_response([0.2] * MOCK_EMBEDDING_DIM)
        mock_query_emb = create_mock_ollama_embedding_response([0.21] * MOCK_EMBEDDING_DIM)
        self.mock_ollama_client_instance.embeddings.side_effect = [mock_dim_emb, mock_text1_emb, mock_query_emb]
        
        tinyresearch._initialize_vector_db(context)
        tinyresearch.add_text_to_vector_db(context, "Test document 1", "source1")
        results = tinyresearch.query_vector_db(context, "query doc 1", top_n=1)
        self.assertIn("Test document 1", results)
        self.assertEqual(self.mock_ollama_client_instance.embeddings.call_count, 3)
        self.mock_litellm_embedding.assert_not_called()

    def test_add_and_query_vector_db_litellm_model(self):
        context = {
            "tokens_used": 0, "embedding_model_name": MOCK_LITELLM_EMBEDDING_MODEL_NAME,
            "gaps_queue": collections.deque(), "knowledge_base": [], "vector_db_texts": []
        }
        mock_dim_emb = create_mock_litellm_embedding_response([[0.1] * MOCK_EMBEDDING_DIM])
        mock_text1_emb = create_mock_litellm_embedding_response([[0.2] * MOCK_EMBEDDING_DIM])
        mock_query_emb = create_mock_litellm_embedding_response([[0.21] * MOCK_EMBEDDING_DIM])
        self.mock_litellm_embedding.side_effect = [mock_dim_emb, mock_text1_emb, mock_query_emb]
        
        tinyresearch._initialize_vector_db(context)
        tinyresearch.add_text_to_vector_db(context, "Test document litellm", "source_litellm")
        results = tinyresearch.query_vector_db(context, "query doc litellm", top_n=1)
        self.assertIn("Test document litellm", results)
        self.assertEqual(self.mock_litellm_embedding.call_count, 3)
        self.mock_ollama_client_instance.embeddings.assert_not_called()

    def test_query_vector_db_empty(self):
        context = {
            "tokens_used": 0, "embedding_model_name": MOCK_OLLAMA_EMBEDDING_MODEL_NAME,
            "gaps_queue": collections.deque()
        }
        # Ensure _get_embedding_dimension uses the ollama mock
        self.mock_ollama_client_instance.embeddings.return_value = create_mock_ollama_embedding_response([0.1] * MOCK_EMBEDDING_DIM)
        tinyresearch._initialize_vector_db(context) # This will call _get_embedding_dimension
        results = tinyresearch.query_vector_db(context, "any query")
        self.assertEqual(results, "No items in vector DB to search.")

    def test_process_user_question_flow(self): 
        mock_args = argparse.Namespace(
            user_question="What is the sky color?", token_budget=10000, max_simulation_steps=1,
            web_browser_user_agent="TestUserAgent", max_url_content_length=1000,
            embedding_model_name=MOCK_OLLAMA_EMBEDDING_MODEL_NAME, # Use Ollama for this flow test
            time_budget_minutes=10, reasoning_style="LLM Default", reasoning_style_active=False,
            llm_provider="litellm", llm_model="ollama/mock-llm", llm_provider_endpoint="http://localhost:11434",
            output_format="text"
        )
        mock_llm_response_content = json.dumps({
            "action_type": "answer",
            "data": {"text": "The sky is blue.", "is_definitive": True, "is_for_original": True}
        })
        self.mock_litellm_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content=mock_llm_response_content))], usage=MagicMock(total_tokens=50))
        
        # Setup Ollama embedding mocks for the flow
        mock_dim_emb_ollama = create_mock_ollama_embedding_response([0.1] * MOCK_EMBEDDING_DIM)
        mock_initial_q_emb_ollama = create_mock_ollama_embedding_response([0.2] * MOCK_EMBEDDING_DIM)
        mock_query_emb_resp_ollama = create_mock_ollama_embedding_response([0.3] * MOCK_EMBEDDING_DIM) # For query_vector_db
        mock_llm_resp_emb_ollama = create_mock_ollama_embedding_response([0.4] * MOCK_EMBEDDING_DIM) # For adding final answer
        self.mock_ollama_client_instance.embeddings.side_effect = [
            mock_dim_emb_ollama, mock_initial_q_emb_ollama, 
            mock_query_emb_resp_ollama, mock_llm_resp_emb_ollama
        ]

        with patch('time.sleep'), patch('tinyresearch.evaluate_answer_with_agent', return_value=True):
            final_result = tinyresearch.run_deep_search_agent(mock_args)
        self.assertEqual(final_result, "The sky is blue.")
        self.mock_litellm_embedding.assert_not_called() # Ensure LiteLLM embedding wasn't used

class TestLLMInteraction(unittest.TestCase):
    def setUp(self):
        self.mock_litellm_completion_patcher = patch('litellm.completion')
        self.mock_genai_patcher = patch('tinyresearch.genai')
        self.mock_ollama_client_patcher = patch('tinyresearch.ollama.Client') 
        self.mock_litellm_embedding_patcher = patch('tinyresearch.litellm.embedding')

        self.mock_litellm_completion = self.mock_litellm_completion_patcher.start()
        self.mock_genai = self.mock_genai_patcher.start()
        self.mock_ollama_client_class = self.mock_ollama_client_patcher.start()
        self.mock_ollama_client_instance = self.mock_ollama_client_class.return_value
        self.mock_litellm_embedding = self.mock_litellm_embedding_patcher.start()
        
        # Default mocks
        self.mock_ollama_client_instance.embeddings.return_value = create_mock_ollama_embedding_response([0.1] * MOCK_EMBEDDING_DIM)
        self.mock_litellm_embedding.return_value = create_mock_litellm_embedding_response([[0.1] * MOCK_EMBEDDING_DIM])

    def tearDown(self):
        self.mock_litellm_completion_patcher.stop()
        self.mock_genai_patcher.stop()
        self.mock_ollama_client_patcher.stop()
        self.mock_litellm_embedding_patcher.stop()

    def test_make_llm_api_call_litellm(self):
        context = {"tokens_used": 0}
        self.mock_litellm_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="LiteLLM says hi"))],
            usage=MagicMock(total_tokens=10)
        )
        response, error = tinyresearch._make_llm_api_call(
            "TestAgent", 
            messages=[{"role": "system", "content": "System prompt"}, {"role": "user", "content": "User prompt"}], # Corrected
            llm_provider="litellm", llm_model="ollama/test", 
            llm_provider_endpoint="http://localhost:11434", max_output_tokens=100, context_for_token_counting=context
        )
        self.assertEqual(response, "LiteLLM says hi")
        self.assertIsNone(error)

    def test_make_llm_api_call_gemini(self):
        context = {"tokens_used": 0}
        tinyresearch.g_gemini_api_key = "test_key"
        mock_gemini_model_instance = MagicMock()
        mock_gemini_response = MagicMock(text="Gemini says hi", usage_metadata=MagicMock(total_token_count=20))
        mock_gemini_model_instance.generate_content.return_value = mock_gemini_response
        self.mock_genai.GenerativeModel.return_value = mock_gemini_model_instance

        response, error = tinyresearch._make_llm_api_call(
            "TestAgent", 
            messages=[{"role": "system", "content": "System prompt"}, {"role": "user", "content": "User prompt"}], # Corrected
            llm_provider="gemini", llm_model="gemini/test", 
            llm_provider_endpoint=None, max_output_tokens=100, context_for_token_counting=context
        )
        self.assertEqual(response, "Gemini says hi")
        self.assertIsNone(error)

    def test_make_llm_api_call_with_memory(self):
        context = {
            "user_question": "Q?", "knowledge_base": [], "gaps_queue": collections.deque(),
            "processed_urls": set(), "processed_files": set(), "tokens_used": 0,
            "embedding_model_name": MOCK_OLLAMA_EMBEDDING_MODEL_NAME, # Use Ollama model for query_vector_db
            "embedding_dim": MOCK_EMBEDDING_DIM,
            "vector_db_index": faiss.IndexFlatL2(MOCK_EMBEDDING_DIM),
            "vector_db_texts": [{"text": "Memory content", "source": "previous_step"}]
        }
        context["vector_db_index"].add(np.array([[0.1] * MOCK_EMBEDDING_DIM], dtype=np.float32))

        # query_vector_db will use the Ollama path due to context's embedding_model_name
        self.mock_ollama_client_instance.embeddings.return_value = create_mock_ollama_embedding_response([0.11] * MOCK_EMBEDDING_DIM) 
        
        self.mock_litellm_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="LLM with memory"))], usage=MagicMock(total_tokens=30))

        memory_str = tinyresearch.query_vector_db(context, "User query")
        self.assertIn("Memory content", memory_str)
        
        messages_with_memory = [ 
            {"role": "system", "content": "System prompt"},
            {"role": "user", "content": f"Memory:\n{memory_str}\n\nUser query"}
        ]
        response, _ = tinyresearch._make_llm_api_call(
            "TestAgent", messages_with_memory, "litellm", "ollama/test", "http://localhost:11434", 100, context
        )
        self.assertEqual(response, "LLM with memory")
        self.mock_litellm_completion.assert_called_once()
        args, kwargs = self.mock_litellm_completion.call_args
        self.assertIn("Memory content", kwargs['messages'][1]['content'])
        self.mock_litellm_embedding.assert_not_called() # Ensure LiteLLM embedding wasn't used for query_vector_db

class TestWebSearch(unittest.TestCase): # Placeholder, assuming similar structure to TestTinyResearchTools
    def setUp(self):
        self.context = {
            "knowledge_base": [], "tokens_used": 0, "urls_to_visit": collections.deque(), 
            "visited_urls": set(), "processed_files": set(),
            "disable_search_next": False, "disable_visit_next": False, "disable_reflect_next": False,
            "embedding_model_name": MOCK_OLLAMA_EMBEDDING_MODEL_NAME, 
            "gaps_queue": collections.deque()
        }
        # Mocking DDGS for context manager - Patching in tinyresearch's namespace
        self.mock_ddgs_patcher = patch('tinyresearch.DDGS') 
        mock_ddgs_class_constructor = self.mock_ddgs_patcher.start() # This is the mock for the DDGS class

        # This is the mock for the instance that DDGS() will produce
        self.mock_ddgs_instance_obj = MagicMock() 
        # This is the mock for the object that `__enter__` will return (the context manager object)
        self.mock_ddgs_cm_obj = MagicMock() 
        self.mock_ddgs_instance_obj.__enter__.return_value = self.mock_ddgs_cm_obj
        mock_ddgs_class_constructor.return_value = self.mock_ddgs_instance_obj

        self.mock_ollama_client_patcher = patch('tinyresearch.ollama.Client')
        self.mock_ollama_client_class = self.mock_ollama_client_patcher.start()
        self.mock_ollama_client_instance = self.mock_ollama_client_class.return_value
        self.mock_ollama_client_instance.embeddings.return_value = create_mock_ollama_embedding_response([0.1] * MOCK_EMBEDDING_DIM)

        self.mock_litellm_embedding_patcher = patch('tinyresearch.litellm.embedding') # Still mock it in case
        self.mock_litellm_embedding = self.mock_litellm_embedding_patcher.start()
        self.mock_litellm_embedding.return_value = create_mock_litellm_embedding_response([[0.1] * MOCK_EMBEDDING_DIM])


    def tearDown(self):
        self.mock_ddgs_patcher.stop()
        self.mock_ollama_client_patcher.stop()
        self.mock_litellm_embedding_patcher.stop()

    def test_perform_web_search_action(self): # Simplified, focuses on execute_search
        mock_search_results = [{'href': 'http://example.com', 'body': 'Mock web content', 'title': 'Mock Title'}]
        # self.mock_ddgs_cm_obj is the object that has the .text() method
        self.mock_ddgs_cm_obj.text.return_value = mock_search_results
    
        mock_dim_emb_ollama = create_mock_ollama_embedding_response([0.1] * MOCK_EMBEDDING_DIM)
        # Provide enough mock embeddings for dimension check + up to MAX_SEARCH_RESULTS snippets
        mock_snippet_embeddings_ollama = [
            create_mock_ollama_embedding_response([0.8 + i*0.01] * MOCK_EMBEDDING_DIM) 
            for i in range(tinyresearch.MAX_SEARCH_RESULTS)
        ]
        self.mock_ollama_client_instance.embeddings.side_effect = [mock_dim_emb_ollama, *mock_snippet_embeddings_ollama]
    
        urls = tinyresearch.execute_search(self.context, "search query")
        
        self.assertIn("http://example.com", urls)
        self.assertEqual(self.context["vector_db_index"].ntotal, 1)
        self.assertIn("Mock web content", self.context["vector_db_texts"][0]["text"])
        self.assertEqual(self.mock_ollama_client_instance.embeddings.call_count, 2)
        self.mock_litellm_embedding.assert_not_called()

class TestFileProcessing(unittest.TestCase): 
    def setUp(self):
        self.mock_ollama_client_patcher = patch('tinyresearch.ollama.Client')
        self.mock_litellm_embedding_patcher = patch('tinyresearch.litellm.embedding')
        self.mock_open_patcher = patch('builtins.open', new_callable=mock_open)
        self.mock_os_path_patcher = patch('os.path')

        self.mock_ollama_client_class = self.mock_ollama_client_patcher.start()
        self.mock_ollama_client_instance = self.mock_ollama_client_class.return_value
        self.mock_litellm_embedding = self.mock_litellm_embedding_patcher.start()
        self.mock_open = self.mock_open_patcher.start()
        self.mock_os_path = self.mock_os_path_patcher.start()

        # Default mocks
        self.mock_ollama_client_instance.embeddings.return_value = create_mock_ollama_embedding_response([0.1] * MOCK_EMBEDDING_DIM)
        self.mock_litellm_embedding.return_value = create_mock_litellm_embedding_response([[0.1] * MOCK_EMBEDDING_DIM])

    def tearDown(self):
        self.mock_ollama_client_patcher.stop()
        self.mock_litellm_embedding_patcher.stop()
        self.mock_open_patcher.stop()
        self.mock_os_path_patcher.stop()

    def test_perform_file_read_action(self): # Simplified, focuses on add_text_to_vector_db
        context = {
            "user_question": "Q?", "knowledge_base": [], "gaps_queue": collections.deque(),
            "processed_urls": set(), "processed_files": set(), "tokens_used": 0,
            "embedding_model_name": MOCK_OLLAMA_EMBEDDING_MODEL_NAME, 
        }
        
        mock_dim_emb_ollama = create_mock_ollama_embedding_response([0.1] * MOCK_EMBEDDING_DIM)
        mock_content_emb_ollama = create_mock_ollama_embedding_response([0.9] * MOCK_EMBEDDING_DIM)
        self.mock_ollama_client_instance.embeddings.side_effect = [mock_dim_emb_ollama, mock_content_emb_ollama]

        self.mock_os_path.exists.return_value = True
        self.mock_os_path.getsize.return_value = 100 
        self.mock_open.return_value.read.return_value = "Mock file content."

        file_path = "test_file.txt"
        
        # Simulate the part of execute_file_read that calls add_text_to_vector_db
        if self.mock_os_path.exists(file_path) and self.mock_os_path.getsize(file_path) < 1000000: # Max file size check
            with open(file_path, 'r', encoding='utf-8') as f: 
                file_content = f.read()
            tinyresearch.add_text_to_vector_db(context, file_content, source_info=f"file_read_{file_path}")
            context["processed_files"].add(file_path)

        self.assertIn("test_file.txt", context["processed_files"])
        self.assertEqual(context["vector_db_index"].ntotal, 1)
        self.assertIn("Mock file content.", context["vector_db_texts"][0]["text"])
        self.assertEqual(self.mock_ollama_client_instance.embeddings.call_count, 2) 
        self.mock_ollama_client_instance.embeddings.assert_any_call(model="mock-embedding-model", prompt="Mock file content.")
        self.mock_litellm_embedding.assert_not_called()

class TestRunDeepSearchAgentLogic(unittest.TestCase):
    def setUp(self):
        self.mock_decide_json_action_patcher = patch('tinyresearch.decide_json_action')
        self.mock_initialize_context_patcher = patch('tinyresearch.initialize_context_and_variables')
        self.mock_get_question_patcher = patch('tinyresearch.get_current_question_from_gaps')
        self.mock_check_budget_patcher = patch('tinyresearch.check_token_budget_exceeded', return_value=False)
        self.mock_check_time_budget_patcher = patch('tinyresearch.check_time_budget_exceeded', return_value=False)
        self.mock_sleep_patcher = patch('time.sleep')
        self.mock_ollama_client_patcher = patch('tinyresearch.ollama.Client') 
        self.mock_litellm_embedding_patcher = patch('tinyresearch.litellm.embedding')

        self.mock_decide_action = self.mock_decide_json_action_patcher.start()
        self.mock_init_context = self.mock_initialize_context_patcher.start()
        self.mock_get_question = self.mock_get_question_patcher.start()
        self.mock_check_budget = self.mock_check_budget_patcher.start()
        self.mock_check_time_budget = self.mock_check_time_budget_patcher.start()
        self.mock_sleep = self.mock_sleep_patcher.start()
        self.mock_ollama_client_class = self.mock_ollama_client_patcher.start()
        self.mock_ollama_client_instance = self.mock_ollama_client_class.return_value
        self.mock_litellm_embedding = self.mock_litellm_embedding_patcher.start()
        
        self.mock_args = argparse.Namespace(
            user_question="Test Q?", token_budget=1000, max_simulation_steps=1,
            web_browser_user_agent="test_agent", max_url_content_length=100,
            llm_provider="litellm", llm_model="test/model", llm_provider_endpoint="http://localhost:11434",
            output_format="text", time_budget_minutes=10,
            reasoning_style="LLM Default", reasoning_style_active=False,
            embedding_model_name=MOCK_OLLAMA_EMBEDDING_MODEL_NAME # Default to Ollama for this test suite
        )
        self.initial_context = {
            "simulation_step": 0, "max_simulation_steps": 1, "gaps_queue": collections.deque(), 
            "knowledge_base": [], "urls_to_visit": collections.deque(), "visited_urls": set(), "processed_files": set(),
            "bad_attempts": [], "known_questions": set(), "tokens_used": 0, "token_budget": 1000, 
            "user_question": "Test Q?", "current_question": None, "start_time": time.time(), 
            "time_budget_seconds": 600, "is_user_question_context": False,
            "disable_reflect_next": False, "disable_search_next": False, "disable_visit_next": False,
            "embedding_model_name": MOCK_OLLAMA_EMBEDDING_MODEL_NAME, "embedding_dim": MOCK_EMBEDDING_DIM,
            "vector_db_index": faiss.IndexFlatL2(MOCK_EMBEDDING_DIM), "vector_db_texts": []
        }
        self.mock_init_context.return_value = self.initial_context
        self.mock_get_question.return_value = {"text": "Test Q?", "is_original": True}
        
        # Mock embeddings for query_vector_db called within run_deep_search_agent
        mock_dim_emb_ollama = create_mock_ollama_embedding_response([0.01] * MOCK_EMBEDDING_DIM)
        mock_q_emb_ollama = create_mock_ollama_embedding_response([0.02] * MOCK_EMBEDDING_DIM)
        # This side_effect will be used by initialize_context_and_variables and then by query_vector_db in the loop
        self.mock_ollama_client_instance.embeddings.side_effect = [
            mock_dim_emb_ollama,  # For _get_embedding_dimension in initialize_context
            mock_q_emb_ollama,    # For add_text_to_vector_db (initial question) in initialize_context
            mock_q_emb_ollama     # For query_vector_db in the main loop
        ]

    def tearDown(self):
        self.mock_decide_json_action_patcher.stop() 
        self.mock_initialize_context_patcher.stop() 
        self.mock_get_question_patcher.stop() 
        self.mock_check_budget_patcher.stop() 
        self.mock_sleep_patcher.stop()
        self.mock_ollama_client_patcher.stop()
        self.mock_litellm_embedding_patcher.stop()
        self.mock_check_time_budget_patcher.stop()

    @patch('tinyresearch.execute_get_stock_price')
    def test_run_deep_search_agent_get_stock_price_action(self, mock_execute_stock):
        self.mock_decide_action.return_value = {"action_type": "get_stock_price", "data": {"ticker": "NVDA"}}
        tinyresearch.run_deep_search_agent(self.mock_args)
        mock_execute_stock.assert_called_once_with(self.initial_context, {"ticker": "NVDA"})

    @patch('tinyresearch.execute_search')
    @patch('tinyresearch.execute_wikipedia_search')
    @patch('tinyresearch.execute_arxiv_search')
    def test_run_deep_search_agent_search_dispatch(self, mock_exec_arxiv, mock_exec_wiki, mock_exec_search):
        with patch('duckduckgo_search.DDGS') as mock_ddgs_local: # Patch DDGS locally for this test
            mock_ddgs_local.return_value.__enter__.return_value.text.return_value = [] 

            self.mock_decide_action.side_effect = [
                {"action_type": "search", "data": {"query": "X"}}, 
                {"action_type": "search_strategy_determined", "data": {"engine_to_use": "wikipedia", "refined_query": "X wiki"}}
            ]
            tinyresearch.run_deep_search_agent(self.mock_args)
            mock_exec_wiki.assert_called_once_with(self.initial_context, "X wiki")

    @patch('tinyresearch.evaluate_answer_with_agent')
    @patch('tinyresearch.store_bad_attempt_reset_context')
    def test_run_deep_search_agent_answer_action_flow_good_answer(self, mock_store_bad_attempt, mock_eval_answer):
        self.mock_decide_action.return_value = {"action_type": "answer", "data": {"text": "Final Answer", "is_definitive": True, "is_for_original": True, "has_references": True}}
        mock_eval_answer.return_value = True
        
        final_answer = tinyresearch.run_deep_search_agent(self.mock_args)
        self.assertEqual(final_answer, "Final Answer")
        mock_eval_answer.assert_called_once()
    
    @patch('tinyresearch.evaluate_answer_with_agent')
    @patch('tinyresearch.store_bad_attempt_reset_context')
    def test_run_deep_search_agent_answer_action_flow_bad_answer_fallback(self, mock_store_bad_attempt, mock_eval_answer):
        self.mock_decide_action.return_value = {"action_type": "answer", "data": {"text": "Bad Answer", "is_definitive": False, "is_for_original": True}}
        mock_eval_answer.return_value = False
        
        with patch('tinyresearch.generate_final_answer_with_agent', return_value="Fallback answer") as mock_fallback:
             tinyresearch.run_deep_search_agent(self.mock_args)
        mock_eval_answer.assert_called_once()
        mock_store_bad_attempt.assert_called_once()
        mock_fallback.assert_called_once()

    @patch('tinyresearch.store_bad_attempt_reset_context')
    def test_run_deep_search_agent_error_action(self, mock_store_bad_attempt):
        self.mock_decide_action.return_value = {"action_type": "error", "data": {"message": "LLM Error", "agent_name": "MainDecisionAgent"}}
        
        with patch('tinyresearch.generate_final_answer_with_agent', return_value="Fallback answer on error") as mock_fallback:
            tinyresearch.run_deep_search_agent(self.mock_args)
        mock_store_bad_attempt.assert_called_once()
        mock_fallback.assert_called_once()

if __name__ == '__main__':
    unittest.main()
