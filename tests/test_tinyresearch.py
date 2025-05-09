import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import json
import time
import argparse
import collections # Import collections
import numpy as np
import faiss
import requests # Import requests
from litellm.utils import ModelResponse, Choices, Message, Usage # For creating mock LiteLLM responses

# Add the parent directory to the Python path to import tinyresearch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tinyresearch

MOCK_EMBEDDING_DIM = 768 # Example dimension for a mock model like nomic-embed-text
MOCK_EMBEDDING_MODEL_NAME = "ollama/mock-embedding-model"

def create_mock_embedding_response(embeddings_list, model_name="mock_embedding_model", prompt_tokens=10, total_tokens=10):
    """
    Helper function to create a mock litellm.embedding response.
    embeddings_list: A list of embedding vectors (each vector is a list of floats).
    """
    data_items = []
    for emb_vector in embeddings_list:
        item = MagicMock()
        item.embedding = emb_vector
        data_items.append(item)
    
    response = ModelResponse(model=model_name, usage=Usage(prompt_tokens=prompt_tokens, total_tokens=total_tokens))
    # Manually set .data as it's not in the constructor in all litellm versions or for all response types
    response.data = data_items 
    # Ensure choices and message are present if some part of litellm expects them, even for embeddings
    response.choices = [Choices(message=Message(content=None), finish_reason="stop")]
    return response

class TestTinyResearchHelpers(unittest.TestCase):

    def test_strip_think_tags(self):
        self.assertEqual(tinyresearch.strip_think_tags("Hello <think>Hmm</think> World"), "Hello World")
        self.assertEqual(tinyresearch.strip_think_tags("No tags here"), "No tags here")
        self.assertEqual(tinyresearch.strip_think_tags("<think>Leading</think> Text"), "Text")
        self.assertEqual(tinyresearch.strip_think_tags("Text <think>Trailing</think>"), "Text")
        self.assertEqual(tinyresearch.strip_think_tags("<think>Only think</think>"), "")
        self.assertEqual(tinyresearch.strip_think_tags("Multiple <think>one</think> tags <think>two</think> here"), "Multiple tags here")
        self.assertEqual(tinyresearch.strip_think_tags("  <think> Spaced </think>  Content  "), "Content")
        self.assertEqual(tinyresearch.strip_think_tags(""), "")

    def test_format_knowledge_snippets_for_prompt(self):
        # Test empty knowledge base
        self.assertEqual(tinyresearch._format_knowledge_snippets_for_prompt([]), "No knowledge gathered yet.")

        # Test stock_data
        kb_stock = [{"type": "stock_data", "data_text": "NVDA: $100"}]
        self.assertIn("- stock_data: NVDA: $100", tinyresearch._format_knowledge_snippets_for_prompt(kb_stock))

        # Test current_datetime
        kb_time = [{"type": "current_datetime", "datetime": "2024-05-08", "timezone": "UTC"}]
        self.assertIn("- current_datetime: 2024-05-08 (Timezone: UTC)", tinyresearch._format_knowledge_snippets_for_prompt(kb_time))

        # Test search_result_snippet
        kb_search = [{"type": "search_result_snippet", "snippet": "This is a search snippet."}]
        self.assertIn("- search_result_snippet: This is a search snippet.", tinyresearch._format_knowledge_snippets_for_prompt(kb_search))

        # Test web_content
        kb_web = [{"type": "web_content", "content": "Web page content here."}]
        self.assertIn("- web_content: Web page content here.", tinyresearch._format_knowledge_snippets_for_prompt(kb_web))

        # Test intermediate_answer
        kb_answer = [{"type": "intermediate_answer", "answer": "An intermediate answer."}]
        self.assertIn("- intermediate_answer: An intermediate answer.", tinyresearch._format_knowledge_snippets_for_prompt(kb_answer))

        # Test generic fallback and truncation
        long_content = "This is very long content that will surely exceed the two hundred character limit for truncation and should be handled gracefully by the formatting function. " * 5 # Make it definitely > 200
        kb_long = [{"type": "other_type", "content": long_content}]
        formatted_long = tinyresearch._format_knowledge_snippets_for_prompt(kb_long)
        self.assertTrue(formatted_long.endswith("..."))
        self.assertLess(len(formatted_long), len(f"- other_type: {long_content}") + 5) # Check it's truncated

        # Test MAX_KNOWLEDGE_SNIPPETS_IN_PROMPT
        original_max_snippets = tinyresearch.MAX_KNOWLEDGE_SNIPPETS_IN_PROMPT
        tinyresearch.MAX_KNOWLEDGE_SNIPPETS_IN_PROMPT = 2 # Temporarily change for test
        kb_many = [
            {"type": "item", "content": "item1"},
            {"type": "item", "content": "item2"},
            {"type": "item", "content": "item3"}
        ]
        formatted_many = tinyresearch._format_knowledge_snippets_for_prompt(kb_many)
        self.assertNotIn("item1", formatted_many)
        self.assertIn("item2", formatted_many)
        self.assertIn("item3", formatted_many)
        tinyresearch.MAX_KNOWLEDGE_SNIPPETS_IN_PROMPT = original_max_snippets # Reset

    def test_process_new_sub_questions(self):
        context_reflect_enabled = {"disable_reflect_next": False}
        context_reflect_disabled = {"disable_reflect_next": True}
        reflect_data = {"sub_questions": ["Q1?", "Q2?"]}

        sub_q_enabled = tinyresearch.process_new_sub_questions(context_reflect_enabled, reflect_data)
        self.assertEqual(sub_q_enabled, ["Q1?", "Q2?"])

        sub_q_disabled = tinyresearch.process_new_sub_questions(context_reflect_disabled, reflect_data)
        self.assertEqual(sub_q_disabled, [])

        self.assertEqual(tinyresearch.process_new_sub_questions(context_reflect_enabled, {}), []) # No sub_questions key

    def test_new_unique_questions_check(self):
        context = {"known_questions": set(["Existing Q1?", "Existing Q2?"])}
        new_questions_all_new = ["New Q1?", "New Q2?"]
        unique = tinyresearch.new_unique_questions_check(context, new_questions_all_new)
        self.assertCountEqual(unique, ["New Q1?", "New Q2?"])
        self.assertIn("New Q1?", context["known_questions"])

        new_questions_mixed = ["Existing Q1?", "New Q3?"]
        unique_mixed = tinyresearch.new_unique_questions_check(context, new_questions_mixed)
        self.assertCountEqual(unique_mixed, ["New Q3?"])
        self.assertIn("New Q3?", context["known_questions"])

        new_questions_all_existing = ["Existing Q1?", "Existing Q2?"]
        unique_existing = tinyresearch.new_unique_questions_check(context, new_questions_all_existing)
        self.assertEqual(unique_existing, [])

    def test_add_to_gaps_queue(self):
        context = {"gaps_queue": []} # Should be a deque in actual code, but list is fine for this test's purpose
        questions_to_add = ["Q1", "Q2"]
        tinyresearch.add_to_gaps_queue(context, questions_to_add)
        self.assertEqual(len(context["gaps_queue"]), 2)
        # Assuming it appends, so order is preserved
        self.assertEqual(context["gaps_queue"][0], {"text": "Q1", "is_original": False})
        self.assertEqual(context["gaps_queue"][1], {"text": "Q2", "is_original": False})

class TestTinyResearchTools(unittest.TestCase):

    def setUp(self):
        self.context = {
            "knowledge_base": [], "tokens_used": 0, "urls_to_visit": [], "visited_urls": set(),
            "disable_search_next": False, "disable_visit_next": False, "disable_reflect_next": False,
            "embedding_model_name": MOCK_EMBEDDING_MODEL_NAME,
            # embedding_dim will be set by _initialize_vector_db if called
        }
        self.mock_litellm_embedding_patcher = patch('tinyresearch.litellm.embedding')
        self.mock_litellm_embedding = self.mock_litellm_embedding_patcher.start()
        # Default return value for any embedding call. Specific tests can override with side_effect.
        self.mock_litellm_embedding.return_value = create_mock_embedding_response(
            [[0.1] * MOCK_EMBEDDING_DIM], prompt_tokens=5, total_tokens=5
        )
        
        self.mock_yfinance_ticker_patcher = patch('tinyresearch.yf.Ticker')
        self.mock_yfinance_ticker = self.mock_yfinance_ticker_patcher.start()

        self.mock_ddgs_patcher = patch('tinyresearch.DDGS')
        self.mock_ddgs_class = self.mock_ddgs_patcher.start()
        self.mock_ddgs_instance = self.mock_ddgs_class.return_value.__enter__.return_value

        self.mock_requests_get_patcher = patch('tinyresearch.requests.get')
        self.mock_requests_get = self.mock_requests_get_patcher.start()

        self.mock_datetime_patcher = patch('tinyresearch.datetime')
        self.mock_datetime_module = self.mock_datetime_patcher.start()

        self.mock_pytz_patcher = patch('tinyresearch.pytz')
        self.mock_pytz_module = self.mock_pytz_patcher.start()

        # Mock for file operations (os.path) if needed by any tool tests
        self.mock_os_path_patcher = patch('os.path')
        self.mock_os_path = self.mock_os_path_patcher.start()


    def tearDown(self):
        self.mock_litellm_embedding_patcher.stop()
        self.mock_yfinance_ticker_patcher.stop()
        self.mock_ddgs_patcher.stop()
        self.mock_requests_get_patcher.stop()
        self.mock_datetime_patcher.stop()
        self.mock_pytz_patcher.stop()
        self.mock_os_path_patcher.stop()

    def test_execute_get_current_datetime(self):
        # Setup mock datetime object and its methods
        mock_now_utc = MagicMock()
        mock_now_utc.astimezone.return_value.strftime.return_value = "2023-01-01 10:00:00 UTC+0000"
        
        self.mock_datetime_module.now.return_value = mock_now_utc
        
        # Setup mock pytz
        mock_utc_tz = MagicMock()
        mock_utc_tz.zone = "UTC"
        self.mock_pytz_module.utc = mock_utc_tz
        self.mock_pytz_module.timezone.return_value = mock_utc_tz # For default case

        result = tinyresearch.execute_get_current_datetime(self.context, {"timezone": "UTC"})
        self.assertEqual(result, "2023-01-01 10:00:00 UTC+0000")
        self.assertEqual(len(self.context["knowledge_base"]), 1)
        self.assertEqual(self.context["knowledge_base"][0]["type"], "current_datetime")
        self.assertEqual(self.context["knowledge_base"][0]["datetime"], "2023-01-01 10:00:00 UTC+0000")
        self.assertEqual(self.context["knowledge_base"][0]["timezone"], "UTC")

        self.context["knowledge_base"] = []
        mock_rome_tz = MagicMock()
        mock_rome_tz.zone = "Europe/Rome"
        self.mock_pytz_module.timezone.return_value = mock_rome_tz
        mock_now_utc.astimezone.return_value.strftime.return_value = "2023-01-01 11:00:00 CET+0100"

        result_rome = tinyresearch.execute_get_current_datetime(self.context, {"timezone": "Europe/Rome"})
        self.assertEqual(result_rome, "2023-01-01 11:00:00 CET+0100")
        self.assertEqual(self.context["knowledge_base"][0]["timezone"], "Europe/Rome")
        self.mock_pytz_module.timezone.assert_called_with("Europe/Rome")

    def test_execute_get_stock_price(self):
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {
            "currentPrice": 150.75, "regularMarketPreviousClose": 148.00,
            "currency": "USD", "symbol": "TEST"
        }
        self.mock_yfinance_ticker.return_value = mock_ticker_instance

        result = tinyresearch.execute_get_stock_price(self.context, {"ticker": "TEST"})
        expected_text = "Stock data for TEST: Current Price: 150.75 USD. Day Change: +1.86%."
        self.assertEqual(result, expected_text)
        self.assertEqual(len(self.context["knowledge_base"]), 1)
        kb_item = self.context["knowledge_base"][0]
        self.assertEqual(kb_item["type"], "stock_data")
        self.assertEqual(kb_item["ticker"], "TEST")
        self.assertEqual(kb_item["data_text"], expected_text)
        self.mock_yfinance_ticker.assert_called_once_with("TEST")

        self.context["knowledge_base"] = []
        result_no_ticker = tinyresearch.execute_get_stock_price(self.context, {})
        self.assertIsNone(result_no_ticker)
        self.assertEqual(len(self.context["knowledge_base"]), 0)

        self.context["knowledge_base"] = []
        self.mock_yfinance_ticker.side_effect = Exception("yfinance API error")
        result_exc = tinyresearch.execute_get_stock_price(self.context, {"ticker": "FAIL"})
        self.assertIsNone(result_exc)
        self.assertEqual(len(self.context["knowledge_base"]), 0)

    def test_execute_search(self):
        mock_search_results = [
            {'title': 'Result 1', 'href': 'http://example.com/1', 'body': 'Snippet 1 for result 1.'},
            {'title': 'Result 2', 'href': 'http://example.com/2', 'body': 'Snippet 2 for result 2.'}
        ]
        self.mock_ddgs_instance.text.return_value = mock_search_results

        # Setup specific side_effect for litellm.embedding for this test
        # 1. Dimension check (if DB not initialized and add_text_to_vector_db calls _initialize_vector_db)
        # 2. Embedding for snippet 1
        # 3. Embedding for snippet 2
        mock_dim_emb = create_mock_embedding_response([[0.1] * MOCK_EMBEDDING_DIM], total_tokens=1)
        mock_snip1_emb = create_mock_embedding_response([[0.2] * MOCK_EMBEDDING_DIM], total_tokens=2)
        mock_snip2_emb = create_mock_embedding_response([[0.3] * MOCK_EMBEDDING_DIM], total_tokens=2)
        # If vector_db_index is not present, _initialize_vector_db will be called first by add_text_to_vector_db
        self.mock_litellm_embedding.side_effect = [mock_dim_emb, mock_snip1_emb, mock_snip2_emb]
        
        # Ensure vector_db_index is not in context to trigger initialization path
        if 'vector_db_index' in self.context: del self.context['vector_db_index']
        if 'embedding_dim' in self.context: del self.context['embedding_dim']


        returned_urls = tinyresearch.execute_search(self.context, "test query")

        self.assertEqual(len(returned_urls), 2)
        self.assertIn('http://example.com/1', returned_urls)
        self.assertEqual(len(self.context["knowledge_base"]), 2) # Snippets added to knowledge_base
        self.assertEqual(self.context["vector_db_index"].ntotal, 2) # Snippets added to vector DB
        self.assertEqual(self.context["knowledge_base"][0]["type"], "search_result_snippet")
        self.assertEqual(self.context["knowledge_base"][0]["snippet"], "Snippet 1 for result 1.")
        self.mock_ddgs_instance.text.assert_called_once_with("test query", max_results=tinyresearch.MAX_SEARCH_RESULTS)
        self.assertEqual(self.mock_litellm_embedding.call_count, 3) # Dim check + 2 snippets

        self.context["knowledge_base"] = []
        self.mock_ddgs_instance.text.side_effect = Exception("Search API error")
        returned_urls_error = tinyresearch.execute_search(self.context, "error query")
        self.assertEqual(len(returned_urls_error), 0)
        self.assertEqual(len(self.context["knowledge_base"]), 0)

    def test_visit_urls(self):
        # Mock requests.get to return a mock response
        mock_response_success = MagicMock(spec=requests.Response) # Use spec for better mock behavior
        mock_response_success.status_code = 200
        long_html_p_tag = b"<p>This is a paragraph of text that is definitely going to be longer than fifty characters to ensure truncation logic is tested.</p>"
        mock_response_success.content = b"<html><head><title>Test Page</title></head><body>" + long_html_p_tag + b"<script>alert('hi')</script></body></html>"
        mock_response_success.raise_for_status.return_value = None
        self.mock_requests_get.return_value = mock_response_success

        self.context["urls_to_visit"] = ["http://example.com/page1"]
        self.context["visited_urls"] = set()
        self.context["web_browser_user_agent"] = "TestAgent/1.0"
        self.context["max_url_content_length"] = 50 # For truncation test

        # Mock BeautifulSoup to return a mock object with get_text
        mock_soup_instance = MagicMock()
        mock_soup_instance.get_text.return_value = "This is a paragraph of text that is definitely going to be longer than fifty characters to ensure truncation logic is tested."
        
        with patch('tinyresearch.BeautifulSoup', return_value=mock_soup_instance) as mock_bs_constructor:
            content_pieces = tinyresearch.visit_urls(self.context, {"urls": ["http://example.com/page1"]})
            mock_bs_constructor.assert_called_once_with(mock_response_success.content, 'html.parser')

        self.assertEqual(len(content_pieces), 1)
        self.assertEqual(content_pieces[0]["url"], "http://example.com/page1")
        self.assertIn("This is a paragraph of text", content_pieces[0]["content"])
        self.assertNotIn("<script>", content_pieces[0]["content"])
        self.assertTrue(content_pieces[0]["content"].endswith("... (content truncated)"))
        self.assertIn("http://example.com/page1", self.context["visited_urls"])
        self.assertNotIn("http://example.com/page1", self.context["urls_to_visit"])
        self.mock_requests_get.assert_called_once_with(
            "http://example.com/page1", headers={'User-Agent': 'TestAgent/1.0'}, timeout=tinyresearch.REQUESTS_TIMEOUT
        )

        content_pieces_again = tinyresearch.visit_urls(self.context, {"urls": ["http://example.com/page1"]})
        self.assertEqual(len(content_pieces_again), 0)

        mock_response_fail = MagicMock(spec=requests.Response) # Add spec back
        self.context["visited_urls"] = set()
        mock_response_fail = MagicMock(spec=requests.Response) # Use spec
        mock_response_fail.raise_for_status.side_effect = tinyresearch.requests.exceptions.HTTPError("403 Forbidden")
        self.mock_requests_get.return_value = mock_response_fail
        
        content_pieces_fail = tinyresearch.visit_urls(self.context, {"urls": ["http://example.com/forbidden"]})
        self.assertEqual(len(content_pieces_fail), 0)
        self.assertNotIn("http://example.com/forbidden", self.context["visited_urls"])

    def test_visit_urls_edge_cases(self):
        self.context["urls_to_visit"] = ["not_a_url", "http://example.com/good"]
        self.context["visited_urls"] = set()
        
        def side_effect_for_get(url, headers, timeout):
            if url == "not_a_url":
                raise tinyresearch.requests.exceptions.RequestException("Invalid URL for test")
            elif url == "http://example.com/good":
                mock_resp = MagicMock(spec=requests.Response) # Use spec
                mock_resp.status_code = 200
                mock_resp.content = b"<p>Good content</p>"
                mock_resp.raise_for_status.return_value = None
                return mock_resp
            raise ValueError(f"Unexpected URL for mock_get in test_visit_urls_edge_cases: {url}")

        mock_soup_instance_good = MagicMock()
        mock_soup_instance_good.get_text.return_value = "Good content"

        self.mock_requests_get.side_effect = side_effect_for_get
        with patch('tinyresearch.BeautifulSoup', return_value=mock_soup_instance_good) as mock_bs_constructor_good:
            content_pieces = tinyresearch.visit_urls(self.context, {"urls": ["not_a_url", "http://example.com/good"]})
            mock_bs_constructor_good.assert_called_once_with(b"<p>Good content</p>", 'html.parser') # Check it was called for the good URL

        self.assertEqual(len(content_pieces), 1)
        self.assertEqual(content_pieces[0]["url"], "http://example.com/good")
        self.assertIn(unittest.mock.call("http://example.com/good", headers=unittest.mock.ANY, timeout=unittest.mock.ANY), self.mock_requests_get.call_args_list)
        self.assertIn(unittest.mock.call("not_a_url", headers=unittest.mock.ANY, timeout=unittest.mock.ANY), self.mock_requests_get.call_args_list)
        self.assertNotIn("not_a_url", self.context["visited_urls"])

class TestTinyResearchContextAndFlow(unittest.TestCase):
    def setUp(self):
        # Common patchers for this test class
        self.mock_requests_get_patcher = patch('requests.get')
        self.mock_ddgs_patcher = patch('tinyresearch.DDGS')
        self.mock_litellm_completion_patcher = patch('litellm.completion')
        self.mock_litellm_embedding_patcher = patch('tinyresearch.litellm.embedding')
        
        self.mock_requests_get = self.mock_requests_get_patcher.start()
        self.mock_ddgs = self.mock_ddgs_patcher.start()
        self.mock_litellm_completion = self.mock_litellm_completion_patcher.start()
        self.mock_litellm_embedding = self.mock_litellm_embedding_patcher.start()

        # Default mock for embedding calls (can be overridden with side_effect in specific tests)
        self.mock_litellm_embedding.return_value = create_mock_embedding_response([[0.1] * MOCK_EMBEDDING_DIM])

        # Mock for file operations
        self.mock_open_patcher = patch('builtins.open', new_callable=unittest.mock.mock_open)
        self.mock_open = self.mock_open_patcher.start()

        self.mock_os_path_patcher = patch('os.path')
        self.mock_os_path = self.mock_os_path_patcher.start()
        self.mock_os_path.exists.return_value = False # Default for file tests

    def tearDown(self):
        self.mock_requests_get_patcher.stop()
        self.mock_ddgs_patcher.stop()
        self.mock_litellm_completion_patcher.stop()
        self.mock_litellm_embedding_patcher.stop()
        self.mock_open_patcher.stop()
        self.mock_os_path_patcher.stop()

    def test_initialize_context_and_variables(self):
        mock_dim_embedding_vector = [0.01] * MOCK_EMBEDDING_DIM
        mock_response_dim_check = create_mock_embedding_response([mock_dim_embedding_vector], prompt_tokens=5, total_tokens=5)
        
        mock_question_embedding_vector = [0.02] * MOCK_EMBEDDING_DIM
        mock_response_add_question = create_mock_embedding_response([mock_question_embedding_vector], prompt_tokens=10, total_tokens=10)

        self.mock_litellm_embedding.side_effect = [
            mock_response_dim_check,
            mock_response_add_question
        ]

        mock_args = argparse.Namespace(
            token_budget=10000, user_question="Test Question?", max_simulation_steps=5,
            llm_provider="litellm", llm_model="ollama/mock-llm",
            web_browser_user_agent="TestUserAgent", max_url_content_length=1000,
            time_budget_minutes=tinyresearch.DEFAULT_TIME_BUDGET_MINUTES,
            reasoning_style="LLM Default", reasoning_style_active=False,
            embedding_model_name=MOCK_EMBEDDING_MODEL_NAME
        )
        # Gaps queue should be a deque
        with patch('collections.deque', side_effect=lambda x=None: collections.deque(x) if x is not None else collections.deque()) as mock_deque_constructor: # Patch collections.deque
            context = tinyresearch.initialize_context_and_variables(mock_args)

        self.assertEqual(context["token_budget"], 10000)
        self.assertEqual(context["user_question"], "Test Question?")
        self.assertIn("Test Question?", context["known_questions"])
        self.assertTrue(context["gaps_queue"]) 
        self.assertEqual(context["gaps_queue"][0]["text"], "Test Question?")
        
        self.assertIsNotNone(context.get('vector_db_index'))
        self.assertEqual(context.get('embedding_dim'), MOCK_EMBEDDING_DIM)
        self.assertEqual(context['vector_db_index'].d, MOCK_EMBEDDING_DIM)
        self.assertEqual(context['vector_db_index'].ntotal, 1)
        self.assertEqual(len(context['vector_db_texts']), 1)
        self.assertEqual(context['vector_db_texts'][0]['text'], "Test Question?")
        self.assertEqual(context['embedding_model_name'], MOCK_EMBEDDING_MODEL_NAME)
        self.assertEqual(context['tokens_used'], 5 + 10)

        self.assertEqual(self.mock_litellm_embedding.call_count, 2)
        call_args_dim = self.mock_litellm_embedding.call_args_list[0]
        self.assertEqual(call_args_dim.kwargs['model'], MOCK_EMBEDDING_MODEL_NAME)
        self.assertEqual(call_args_dim.kwargs['input'], ["get dimension test string"])
        call_args_q = self.mock_litellm_embedding.call_args_list[1]
        self.assertEqual(call_args_q.kwargs['model'], MOCK_EMBEDDING_MODEL_NAME)
        self.assertEqual(call_args_q.kwargs['input'], ["Test Question?"])

    def test_add_and_query_vector_db(self):
        context = {
            "user_question": "Initial question for context", "knowledge_base": [],
            "gaps_queue": collections.deque(), # Use actual deque
            "processed_urls": set(), "processed_files": set(), "tokens_used": 0,
            "embedding_model_name": MOCK_EMBEDDING_MODEL_NAME,
        }

        mock_dim_emb = [0.1] * MOCK_EMBEDDING_DIM
        mock_text1_emb = [0.2] * MOCK_EMBEDDING_DIM
        mock_text2_emb = [0.3] * MOCK_EMBEDDING_DIM
        mock_query_emb = [0.21] * MOCK_EMBEDDING_DIM

        self.mock_litellm_embedding.side_effect = [
            create_mock_embedding_response([mock_dim_emb], total_tokens=5),
            create_mock_embedding_response([mock_text1_emb], total_tokens=10),
            create_mock_embedding_response([mock_text2_emb], total_tokens=10),
            create_mock_embedding_response([mock_query_emb], total_tokens=8)
        ]

        tinyresearch._initialize_vector_db(context) 
        self.assertEqual(context.get('embedding_dim'), MOCK_EMBEDDING_DIM)
        self.assertIsNotNone(context.get('vector_db_index'))
        self.assertEqual(context['vector_db_index'].d, MOCK_EMBEDDING_DIM)
        self.assertEqual(context['tokens_used'], 5)

        text1 = "This is the first document."
        text2 = "Another document with different content."

        tinyresearch.add_text_to_vector_db(context, text1, "source1")
        self.assertEqual(context['vector_db_index'].ntotal, 1)
        self.assertEqual(len(context['vector_db_texts']), 1)
        self.assertEqual(context['vector_db_texts'][0]['text'], text1)
        self.assertEqual(context['tokens_used'], 5 + 10)

        tinyresearch.add_text_to_vector_db(context, text2, "source2")
        self.assertEqual(context['vector_db_index'].ntotal, 2)
        self.assertEqual(len(context['vector_db_texts']), 2)
        self.assertEqual(context['vector_db_texts'][1]['text'], text2)
        self.assertEqual(context['tokens_used'], 5 + 10 + 10)

        query = "first document query"
        results = tinyresearch.query_vector_db(context, query, top_n=1)
        self.assertIn("Source: source1", results)
        self.assertIn(text1[:50], results)
        self.assertNotIn("source2", results)
        self.assertEqual(context['tokens_used'], 5 + 10 + 10 + 8)
        self.assertEqual(self.mock_litellm_embedding.call_count, 4)

    def test_query_vector_db_empty(self):
        context = {
            "tokens_used": 0, "embedding_model_name": MOCK_EMBEDDING_MODEL_NAME,
            "gaps_queue": collections.deque() # Add for completeness if any function uses it
        }
        self.mock_litellm_embedding.return_value = create_mock_embedding_response([[0.1] * MOCK_EMBEDDING_DIM])
        
        tinyresearch._initialize_vector_db(context)
        self.assertEqual(context['vector_db_index'].ntotal, 0)
        results = tinyresearch.query_vector_db(context, "any query") # This call will now return "No items in vector DB to search."
        self.assertEqual(results, "No items in vector DB to search.") # Corrected expected value

    def test_process_user_question_flow(self):
        mock_args = argparse.Namespace(
            user_question="What is the sky color?", token_budget=10000, max_simulation_steps=1,
            web_browser_user_agent="TestUserAgent", max_url_content_length=1000,
            embedding_model_name=MOCK_EMBEDDING_MODEL_NAME,
            time_budget_minutes=tinyresearch.DEFAULT_TIME_BUDGET_MINUTES,
            reasoning_style="LLM Default", reasoning_style_active=False,
            llm_provider="litellm", llm_model="ollama/mock-llm", llm_provider_endpoint="http://localhost:11434",
            output_format="text" # Added for main
        )
        # Mock LLM to return a valid JSON action
        mock_llm_response_content = json.dumps({
            "action_type": "answer",
            "data": {"text": "The sky is blue based on current knowledge.", "is_definitive": True, "is_for_original": True, "has_references": False}
        })

        self.mock_litellm_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content=mock_llm_response_content))], usage=MagicMock(total_tokens=50))
        
        mock_dim_emb = create_mock_embedding_response([[0.1] * MOCK_EMBEDDING_DIM])
        mock_initial_q_emb = create_mock_embedding_response([[0.2] * MOCK_EMBEDDING_DIM])
        mock_query_emb_resp = create_mock_embedding_response([[0.3] * MOCK_EMBEDDING_DIM]) 
        mock_llm_resp_emb = create_mock_embedding_response([[0.4] * MOCK_EMBEDDING_DIM]) # For adding final answer

        self.mock_litellm_embedding.side_effect = [mock_dim_emb, mock_initial_q_emb, mock_query_emb_resp, mock_llm_resp_emb]

        # Patch time.sleep to avoid delays
        with patch('time.sleep'):
            final_context_or_result = tinyresearch.run_deep_search_agent(mock_args)

        self.mock_litellm_completion.assert_called()
        self.assertIsNotNone(final_context_or_result)
        # The final result when max steps is reached is a fallback message, not the LLM's answer
        self.assertIn("max steps (1) reached", final_context_or_result) # Check for the fallback message

    def test_process_web_search_result(self):
        context = {
            "user_question": "Q", "knowledge_base": [], "gaps_queue": collections.deque(),
            "processed_urls": set(), "processed_files": set(), "tokens_used": 0,
            "embedding_model_name": MOCK_EMBEDDING_MODEL_NAME,
            # embedding_dim and vector_db_index will be initialized by add_text_to_vector_db
        }
        
        # Embedding calls: 1 for dim check, 1 for the content
        mock_dim_emb = create_mock_embedding_response([[0.1] * MOCK_EMBEDDING_DIM], total_tokens=1)
        mock_content_emb = create_mock_embedding_response([[0.6] * MOCK_EMBEDDING_DIM], total_tokens=5)
        self.mock_litellm_embedding.side_effect = [mock_dim_emb, mock_content_emb]

        web_content = "Useful information from the web."
        tinyresearch.add_text_to_vector_db(context, web_content, source_info="web_search_http://example.com/info")
        context["processed_urls"].add("http://example.com/info")

        self.assertIn("http://example.com/info", context["processed_urls"])
        self.assertEqual(context["vector_db_index"].ntotal, 1)
        self.assertIn(web_content, context["vector_db_texts"][0]["text"])
        self.assertEqual(self.mock_litellm_embedding.call_count, 2)
        self.mock_litellm_embedding.assert_any_call(model=MOCK_EMBEDDING_MODEL_NAME, input=[web_content], api_base='http://localhost:11434') # Expect api_base

    def test_process_file_content(self):
        context = {
            "user_question": "Q", "knowledge_base": [], "gaps_queue": collections.deque(),
            "processed_urls": set(), "processed_files": set(), "tokens_used": 0,
            "embedding_model_name": MOCK_EMBEDDING_MODEL_NAME,
            # embedding_dim and vector_db_index will be initialized by add_text_to_vector_db
        }
        
        # Embedding calls: 1 for dim check, 1 for the content
        mock_dim_emb = create_mock_embedding_response([[0.1] * MOCK_EMBEDDING_DIM], total_tokens=1)
        mock_content_emb = create_mock_embedding_response([[0.9] * MOCK_EMBEDDING_DIM], total_tokens=5)
        self.mock_litellm_embedding.side_effect = [mock_dim_emb, mock_content_emb]
        
        file_content = "Content from a local file."
        tinyresearch.add_text_to_vector_db(context, file_content, source_info="file_read_test.txt")
        context["processed_files"].add("test.txt")

        self.assertIn("test.txt", context["processed_files"])
        self.assertEqual(context["vector_db_index"].ntotal, 1)
        self.assertIn(file_content, context["vector_db_texts"][0]["text"])
        self.assertEqual(self.mock_litellm_embedding.call_count, 2)
        self.mock_litellm_embedding.assert_any_call(model=MOCK_EMBEDDING_MODEL_NAME, input=[file_content], api_base='http://localhost:11434') # Expect api_base

class TestLLMInteraction(unittest.TestCase):
    def setUp(self):
        self.mock_litellm_completion_patcher = patch('litellm.completion')
        self.mock_litellm_embedding_patcher = patch('tinyresearch.litellm.embedding')
        self.mock_genai_patcher = patch('tinyresearch.genai')


        self.mock_litellm_completion = self.mock_litellm_completion_patcher.start()
        self.mock_litellm_embedding = self.mock_litellm_embedding_patcher.start()
        self.mock_genai = self.mock_genai_patcher.start()


        self.mock_litellm_embedding.return_value = create_mock_embedding_response([[0.1] * MOCK_EMBEDDING_DIM])

    def tearDown(self):
        self.mock_litellm_completion_patcher.stop()
        self.mock_litellm_embedding_patcher.stop()
        self.mock_genai_patcher.stop()


    def test_make_llm_api_call_litellm(self):
        context = {"tokens_used": 0}
        self.mock_litellm_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="LiteLLM says hi"))],
            usage=MagicMock(total_tokens=10)
        )
        response, error = tinyresearch._make_llm_api_call(
            "TestAgent", "System prompt", "User prompt", "litellm", "ollama/test", "http://localhost:11434", 100, context
        )
        self.assertEqual(response, "LiteLLM says hi")
        self.assertIsNone(error)
        self.assertEqual(context["tokens_used"], 10)
        self.mock_litellm_completion.assert_called_once()
        call_args = self.mock_litellm_completion.call_args[1] # kwargs
        self.assertEqual(call_args['model'], "ollama/test")
        self.assertEqual(call_args['api_base'], "http://localhost:11434")

    def test_make_llm_api_call_gemini(self):
        context = {"tokens_used": 0}
        tinyresearch.g_gemini_api_key = "test_key" # Ensure API key is set for Gemini
        
        mock_gemini_model_instance = MagicMock()
        mock_gemini_response = MagicMock(
            text="Gemini says hi",
            usage_metadata=MagicMock(total_token_count=20)
        )
        mock_gemini_model_instance.generate_content.return_value = mock_gemini_response
        self.mock_genai.GenerativeModel.return_value = mock_gemini_model_instance

        response, error = tinyresearch._make_llm_api_call(
            "TestAgent", "System prompt", "User prompt", "gemini", "gemini/test", None, 100, context
        )
        self.assertEqual(response, "Gemini says hi")
        self.assertIsNone(error)
        self.assertEqual(context["tokens_used"], 20)
        self.mock_genai.GenerativeModel.assert_called_once_with("gemini/test")
        mock_gemini_model_instance.generate_content.assert_called_once()

    def test_make_llm_api_call_with_memory(self):
        context = {
            "user_question": "Q?", "knowledge_base": [], "gaps_queue": collections.deque(),
            "processed_urls": set(), "processed_files": set(), "tokens_used": 0,
            "embedding_model_name": MOCK_EMBEDDING_MODEL_NAME, "embedding_dim": MOCK_EMBEDDING_DIM,
            "vector_db_index": faiss.IndexFlatL2(MOCK_EMBEDDING_DIM),
            "vector_db_texts": [{"text": "Memory content", "source": "previous_step"}]
        }
        context["vector_db_index"].add(np.array([[0.1] * MOCK_EMBEDDING_DIM], dtype=np.float32))

        self.mock_litellm_embedding.return_value = create_mock_embedding_response([[0.11] * MOCK_EMBEDDING_DIM]) # For query
        self.mock_litellm_completion.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="LLM with memory"))], usage=MagicMock(total_tokens=30))

        memory_str = tinyresearch.query_vector_db(context, "User query")
        self.assertIn("Memory content", memory_str)
        
        full_prompt = f"System prompt\n\nMemory:\n{memory_str}\n\nUser query"
        response, _ = tinyresearch._make_llm_api_call(
            "TestAgent", "System prompt", full_prompt, "litellm", "ollama/test", "http://localhost:11434", 100, context
        )
        
        self.assertEqual(response, "LLM with memory")
        self.mock_litellm_completion.assert_called_once()
        args, kwargs = self.mock_litellm_completion.call_args
        self.assertIn("Memory content", kwargs['messages'][1]['content'])

class TestWebSearch(unittest.TestCase):
    def setUp(self):
        self.mock_ddgs_patcher = patch('tinyresearch.DDGS')
        self.mock_litellm_embedding_patcher = patch('tinyresearch.litellm.embedding')
        
        self.mock_ddgs_class = self.mock_ddgs_patcher.start()
        self.mock_ddgs_instance = self.mock_ddgs_class.return_value.__enter__.return_value
        self.mock_litellm_embedding = self.mock_litellm_embedding_patcher.start()
        
        self.mock_litellm_embedding.return_value = create_mock_embedding_response([[0.1] * MOCK_EMBEDDING_DIM])

    def tearDown(self):
        self.mock_ddgs_patcher.stop()
        self.mock_litellm_embedding_patcher.stop()

    def test_perform_web_search_action(self):
        context = {
            "user_question": "Q?", "knowledge_base": [], "gaps_queue": collections.deque(),
            "processed_urls": set(), "processed_files": set(), "tokens_used": 0,
            "embedding_model_name": MOCK_EMBEDDING_MODEL_NAME,
            "disable_search_next": False, # Added to prevent KeyError if logic checks it
            # No vector_db_index, so it will be initialized
        }
        
        mock_search_results = [{'href': 'http://example.com', 'body': 'Mock web content', 'title': 'Mock Title'}]
        self.mock_ddgs_instance.text.return_value = mock_search_results

        mock_dim_emb = create_mock_embedding_response([[0.1] * MOCK_EMBEDDING_DIM])
        mock_content_emb = create_mock_embedding_response([[0.8] * MOCK_EMBEDDING_DIM])
        self.mock_litellm_embedding.side_effect = [mock_dim_emb, mock_content_emb]

        urls = tinyresearch.execute_search(context, "search query")
        
        self.assertIn("http://example.com", urls)
        self.assertEqual(context["vector_db_index"].ntotal, 1)
        self.assertIn("Mock web content", context["vector_db_texts"][0]["text"])
        self.assertEqual(self.mock_litellm_embedding.call_count, 2)
        self.mock_litellm_embedding.assert_any_call(model=MOCK_EMBEDDING_MODEL_NAME, input=["Mock Title Mock web content"], api_base='http://localhost:11434') # Expect api_base


class TestFileProcessing(unittest.TestCase):
    def setUp(self):
        self.mock_litellm_embedding_patcher = patch('tinyresearch.litellm.embedding')
        self.mock_open_patcher = patch('builtins.open', new_callable=unittest.mock.mock_open)
        self.mock_os_path_patcher = patch('os.path')

        self.mock_litellm_embedding = self.mock_litellm_embedding_patcher.start()
        self.mock_open = self.mock_open_patcher.start()
        self.mock_os_path = self.mock_os_path_patcher.start()

        self.mock_litellm_embedding.return_value = create_mock_embedding_response([[0.1] * MOCK_EMBEDDING_DIM])

    def tearDown(self):
        self.mock_litellm_embedding_patcher.stop()
        self.mock_open_patcher.stop()
        self.mock_os_path_patcher.stop()

    def test_perform_file_read_action(self):
        context = {
            "user_question": "Q?", "knowledge_base": [], "gaps_queue": collections.deque(),
            "processed_urls": set(), "processed_files": set(), "tokens_used": 0,
            "embedding_model_name": MOCK_EMBEDDING_MODEL_NAME,
            # No vector_db_index, so it will be initialized
        }
        
        mock_dim_emb = create_mock_embedding_response([[0.1] * MOCK_EMBEDDING_DIM], total_tokens=1)
        mock_content_emb = create_mock_embedding_response([[0.9] * MOCK_EMBEDDING_DIM], total_tokens=5)
        self.mock_litellm_embedding.side_effect = [mock_dim_emb, mock_content_emb]

        self.mock_os_path.exists.return_value = True
        self.mock_os_path.getsize.return_value = 100 # Small file
        self.mock_open.return_value.read.return_value = "Mock file content."

        file_path = "test_file.txt"
        if self.mock_os_path.exists(file_path) and self.mock_os_path.getsize(file_path) < 1000000:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            tinyresearch.add_text_to_vector_db(context, file_content, source_info=f"file_read_{file_path}")
            context["processed_files"].add(file_path)

        self.assertIn("test_file.txt", context["processed_files"])
        self.assertEqual(context["vector_db_index"].ntotal, 1)
        self.assertIn("Mock file content.", context["vector_db_texts"][0]["text"])
        self.assertEqual(self.mock_litellm_embedding.call_count, 2)
        self.mock_litellm_embedding.assert_any_call(model=MOCK_EMBEDDING_MODEL_NAME, input=["Mock file content."], api_base='http://localhost:11434') # Expect api_base

class TestRunDeepSearchAgentLogic(unittest.TestCase):
    def setUp(self):
        self.mock_decide_json_action_patcher = patch('tinyresearch.decide_json_action')
        self.mock_initialize_context_patcher = patch('tinyresearch.initialize_context_and_variables')
        self.mock_get_question_patcher = patch('tinyresearch.get_current_question_from_gaps')
        self.mock_check_budget_patcher = patch('tinyresearch.check_token_budget_exceeded', return_value=False)
        self.mock_check_time_budget_patcher = patch('tinyresearch.check_time_budget_exceeded', return_value=False)
        self.mock_sleep_patcher = patch('time.sleep')
        self.mock_litellm_embedding_patcher = patch('tinyresearch.litellm.embedding')

        self.mock_decide_action = self.mock_decide_json_action_patcher.start()
        self.mock_init_context = self.mock_initialize_context_patcher.start()
        self.mock_get_question = self.mock_get_question_patcher.start()
        self.mock_check_budget = self.mock_check_budget_patcher.start()
        self.mock_check_time_budget = self.mock_check_time_budget_patcher.start()
        self.mock_sleep = self.mock_sleep_patcher.start()
        self.mock_litellm_embedding = self.mock_litellm_embedding_patcher.start()
        
        self.mock_args = argparse.Namespace(
            user_question="Test Q?", token_budget=1000, max_simulation_steps=1,
            web_browser_user_agent="test_agent", max_url_content_length=100,
            llm_provider="litellm", llm_model="test/model", llm_provider_endpoint="http://localhost:11434",
            output_format="text", time_budget_minutes=10,
            reasoning_style="LLM Default", reasoning_style_active=False,
            embedding_model_name=MOCK_EMBEDDING_MODEL_NAME
        )
        self.initial_context = {
            "simulation_step": 0, "max_simulation_steps": 1, "gaps_queue": collections.deque(), 
            "knowledge_base": [], "urls_to_visit": [], "visited_urls": set(), 
            "bad_attempts": [], "known_questions": set(), "tokens_used": 0, "token_budget": 1000, 
            "user_question": "Test Q?", "current_question": None, "start_time": time.time(), 
            "time_budget_seconds": 600, "is_user_question_context": False,
            "disable_reflect_next": False, "disable_search_next": False, "disable_visit_next": False,
            "embedding_model_name": MOCK_EMBEDDING_MODEL_NAME, "embedding_dim": MOCK_EMBEDDING_DIM,
            "vector_db_index": faiss.IndexFlatL2(MOCK_EMBEDDING_DIM), "vector_db_texts": []
        }
        self.mock_init_context.return_value = self.initial_context
        self.mock_get_question.return_value = {"text": "Test Q?", "is_original": True}
        mock_dim_emb = create_mock_embedding_response([[0.01] * MOCK_EMBEDDING_DIM])
        mock_q_emb = create_mock_embedding_response([[0.02] * MOCK_EMBEDDING_DIM])
        self.mock_litellm_embedding.side_effect = [mock_dim_emb, mock_q_emb, mock_dim_emb, mock_q_emb]

    def tearDown(self):
        # Stop patchers explicitly started in setUp
        self.mock_decide_json_action_patcher.stop() 
        self.mock_initialize_context_patcher.stop() 
        self.mock_get_question_patcher.stop() 
        self.mock_check_budget_patcher.stop() 
        self.mock_sleep_patcher.stop()
        self.mock_litellm_embedding_patcher.stop()
        self.mock_check_time_budget_patcher.stop() # Stop the time budget patcher

    @patch('tinyresearch.execute_get_stock_price')
    def test_run_deep_search_agent_get_stock_price_action(self, mock_execute_stock):
        self.mock_decide_action.return_value = {"action_type": "get_stock_price", "data": {"ticker": "NVDA"}}
        tinyresearch.run_deep_search_agent(self.mock_args)
        mock_execute_stock.assert_called_once_with(self.initial_context, {"ticker": "NVDA"})

    @patch('tinyresearch.execute_search')
    @patch('tinyresearch.execute_wikipedia_search')
    @patch('tinyresearch.execute_arxiv_search')
    def test_run_deep_search_agent_search_dispatch(self, mock_exec_arxiv, mock_exec_wiki, mock_exec_search):
        self.mock_decide_action.side_effect = [
            {"action_type": "search", "data": {"query": "X"}}, 
            {"action_type": "search_strategy_determined", "data": {"engine_to_use": "wikipedia", "refined_query": "X wiki"}}
        ]
        tinyresearch.run_deep_search_agent(self.mock_args)
        mock_exec_wiki.assert_called_once_with(self.initial_context, "X wiki")
        mock_exec_search.assert_not_called()
        mock_exec_arxiv.assert_not_called()

        mock_exec_wiki.reset_mock(); mock_exec_search.reset_mock(); mock_exec_arxiv.reset_mock()
        self.initial_context["simulation_step"] = 0

        self.mock_decide_action.side_effect = [
            {"action_type": "search", "data": {"query": "Y"}},
            {"action_type": "search_strategy_determined", "data": {"engine_to_use": "arxiv", "refined_query": "Y arxiv"}}
        ]
        tinyresearch.run_deep_search_agent(self.mock_args)
        mock_exec_arxiv.assert_called_once_with(self.initial_context, "Y arxiv")

    # Split the answer action flow test into two
    @patch('tinyresearch.evaluate_answer_with_agent')
    @patch('tinyresearch.store_bad_attempt_reset_context')
    def test_run_deep_search_agent_answer_action_flow_good_answer(self, mock_store_bad_attempt, mock_eval_answer):
        self.mock_decide_action.return_value = {"action_type": "answer", "data": {"text": "Final Answer", "is_definitive": True, "is_for_original": True, "has_references": True}}
        mock_eval_answer.return_value = True
        
        final_answer = tinyresearch.run_deep_search_agent(self.mock_args)
        self.assertEqual(final_answer, "Final Answer") # This should now pass with the tinyresearch.py fix
        mock_eval_answer.assert_called_once()
        mock_store_bad_attempt.assert_not_called()
    
    @patch('tinyresearch.evaluate_answer_with_agent')
    @patch('tinyresearch.store_bad_attempt_reset_context')
    def test_run_deep_search_agent_answer_action_flow_bad_answer_fallback(self, mock_store_bad_attempt, mock_eval_answer):
        # Scenario 2: Bad answer
        self.mock_decide_action.return_value = {"action_type": "answer", "data": {"text": "Bad Answer", "is_definitive": False, "is_for_original": True}}
        mock_eval_answer.return_value = False
        
        with patch('tinyresearch.generate_final_answer_with_agent', return_value="Fallback answer") as mock_fallback:
             tinyresearch.run_deep_search_agent(self.mock_args)
        mock_eval_answer.assert_called_once()
        mock_store_bad_attempt.assert_called_once()
        # The fallback answer is generated and returned when max steps is reached
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
