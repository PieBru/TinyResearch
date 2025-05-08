import unittest
from unittest.mock import patch, MagicMock
import sys
import os
import json
import time # Added for time.time() in other tests
import argparse # For testing initialize_context_and_variables
import numpy as np # For mocking sentence_transformers.encode output

# Add the parent directory to the Python path to import tinyresearch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tinyresearch

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
        context = {"gaps_queue": []}
        questions_to_add = ["Q1", "Q2"]
        tinyresearch.add_to_gaps_queue(context, questions_to_add)
        self.assertEqual(len(context["gaps_queue"]), 2)
        self.assertEqual(context["gaps_queue"][0], {"text": "Q1", "is_original": False})
        self.assertEqual(context["gaps_queue"][1], {"text": "Q2", "is_original": False})


class TestTinyResearchTools(unittest.TestCase):

    def setUp(self):
        self.context = {
            "knowledge_base": [],
            "tokens_used": 0, # Required for _make_llm_api_call if we were testing it directly
            "urls_to_visit": [],
            "visited_urls": set(),
            "disable_search_next": False, # Add missing key
            "disable_visit_next": False,  # Add missing key
            "disable_reflect_next": False # Add for completeness, though not directly tested here yet

        }

    @patch('tinyresearch.datetime') # Mocking the datetime object from tinyresearch's scope
    @patch('tinyresearch.pytz')    # Mocking the pytz object from tinyresearch's scope
    def test_execute_get_current_datetime(self, mock_pytz, mock_datetime_module):
        # Setup mock datetime object and its methods
        mock_now_utc = MagicMock()
        mock_now_utc.astimezone.return_value.strftime.return_value = "2023-01-01 10:00:00 UTC+0000"
        # mock_now_utc.astimezone.return_value.zone = "UTC" # .zone on datetime object is not standard
        
        mock_datetime_module.now.return_value = mock_now_utc
        
        # Setup mock pytz
        mock_utc_tz = MagicMock()
        mock_utc_tz.zone = "UTC" # This is what we use in the code to get the string name
        mock_pytz.utc = mock_utc_tz
        mock_pytz.timezone.return_value = mock_utc_tz # For default case

        # Test with default UTC
        result = tinyresearch.execute_get_current_datetime(self.context, {"timezone": "UTC"})
        self.assertEqual(result, "2023-01-01 10:00:00 UTC+0000")
        self.assertEqual(len(self.context["knowledge_base"]), 1)
        self.assertEqual(self.context["knowledge_base"][0]["type"], "current_datetime")
        self.assertEqual(self.context["knowledge_base"][0]["datetime"], "2023-01-01 10:00:00 UTC+0000")
        self.assertEqual(self.context["knowledge_base"][0]["timezone"], "UTC") # Asserting against the .zone attribute

        # Reset knowledge base for next test case within this method
        self.context["knowledge_base"] = []

        # Test with a specific timezone
        mock_rome_tz = MagicMock()
        mock_rome_tz.zone = "Europe/Rome"
        mock_pytz.timezone.return_value = mock_rome_tz # Simulate pytz.timezone("Europe/Rome")
        mock_now_utc.astimezone.return_value.strftime.return_value = "2023-01-01 11:00:00 CET+0100"
        # mock_now_utc.astimezone.return_value.zone = "Europe/Rome" # Not needed as we use mock_rome_tz.zone

        result_rome = tinyresearch.execute_get_current_datetime(self.context, {"timezone": "Europe/Rome"})
        self.assertEqual(result_rome, "2023-01-01 11:00:00 CET+0100")
        self.assertEqual(self.context["knowledge_base"][0]["timezone"], "Europe/Rome")
        mock_pytz.timezone.assert_called_with("Europe/Rome")

    @patch('tinyresearch.yf.Ticker') # Mock yfinance.Ticker
    def test_execute_get_stock_price(self, mock_yfinance_ticker):
        # Setup mock Ticker instance and its .info attribute
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.info = {
            "currentPrice": 150.75,
            "regularMarketPreviousClose": 148.00,
            "currency": "USD",
            "symbol": "TEST"
        }
        mock_yfinance_ticker.return_value = mock_ticker_instance

        result = tinyresearch.execute_get_stock_price(self.context, {"ticker": "TEST"})
        expected_text = "Stock data for TEST: Current Price: 150.75 USD. Day Change: +1.86%."
        self.assertEqual(result, expected_text)
        self.assertEqual(len(self.context["knowledge_base"]), 1)
        kb_item = self.context["knowledge_base"][0]
        self.assertEqual(kb_item["type"], "stock_data")
        self.assertEqual(kb_item["ticker"], "TEST")
        self.assertEqual(kb_item["data_text"], expected_text)
        mock_yfinance_ticker.assert_called_once_with("TEST")

        # Test case where ticker is not provided
        self.context["knowledge_base"] = [] # Reset
        result_no_ticker = tinyresearch.execute_get_stock_price(self.context, {})
        self.assertIsNone(result_no_ticker)
        self.assertEqual(len(self.context["knowledge_base"]), 0)

        # Test yfinance exception
        self.context["knowledge_base"] = [] # Reset
        mock_yfinance_ticker.side_effect = Exception("yfinance API error")
        result_exc = tinyresearch.execute_get_stock_price(self.context, {"ticker": "FAIL"})
        self.assertIsNone(result_exc)
        self.assertEqual(len(self.context["knowledge_base"]), 0)

    @patch('tinyresearch.DDGS')
    def test_execute_search(self, mock_ddgs_class):
        mock_ddgs_instance = MagicMock()
        mock_search_results = [
            {'title': 'Result 1', 'href': 'http://example.com/1', 'body': 'Snippet 1 for result 1.'},
            {'title': 'Result 2', 'href': 'http://example.com/2', 'body': 'Snippet 2 for result 2.'}
        ]
        mock_ddgs_instance.text.return_value = mock_search_results
        mock_ddgs_class.return_value.__enter__.return_value = mock_ddgs_instance # Handle context manager

        returned_urls = tinyresearch.execute_search(self.context, "test query")

        self.assertEqual(len(returned_urls), 2)
        self.assertIn('http://example.com/1', returned_urls)
        self.assertEqual(len(self.context["knowledge_base"]), 2)
        self.assertEqual(self.context["knowledge_base"][0]["type"], "search_result_snippet")
        self.assertEqual(self.context["knowledge_base"][0]["snippet"], "Snippet 1 for result 1.")
        mock_ddgs_instance.text.assert_called_once_with("test query", max_results=tinyresearch.MAX_SEARCH_RESULTS)

        # Test search error
        self.context["knowledge_base"] = [] # Reset
        mock_ddgs_instance.text.side_effect = Exception("Search API error")
        returned_urls_error = tinyresearch.execute_search(self.context, "error query")
        self.assertEqual(len(returned_urls_error), 0)
        self.assertEqual(len(self.context["knowledge_base"]), 0)

    @patch('tinyresearch.requests.get')
    def test_visit_urls(self, mock_requests_get):
        # Mock successful visit
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        # Make the content long enough to be truncated
        long_html_p_tag = b"<p>This is a paragraph of text that is definitely going to be longer than fifty characters to ensure truncation logic is tested.</p>"
        mock_response_success.content = b"<html><head><title>Test Page</title></head><body>" + long_html_p_tag + b"<script>alert('hi')</script></body></html>"

        mock_response_success.raise_for_status.return_value = None # Simulate no HTTPError
        mock_requests_get.return_value = mock_response_success

        self.context["urls_to_visit"] = ["http://example.com/page1"]
        self.context["visited_urls"] = set()
        self.context["web_browser_user_agent"] = "TestAgent/1.0"
        self.context["max_url_content_length"] = 50

        content_pieces = tinyresearch.visit_urls(self.context, {"urls": ["http://example.com/page1"]})

        self.assertEqual(len(content_pieces), 1)
        self.assertEqual(content_pieces[0]["url"], "http://example.com/page1")
        self.assertIn("This is a paragraph of text", content_pieces[0]["content"]) # Check for part of the new content
        self.assertNotIn("<script>", content_pieces[0]["content"]) # Script tags should be removed
        self.assertTrue(content_pieces[0]["content"].endswith("... (content truncated)")) # Check truncation
        self.assertIn("http://example.com/page1", self.context["visited_urls"])
        self.assertNotIn("http://example.com/page1", self.context["urls_to_visit"]) # Should be removed
        mock_requests_get.assert_called_once_with(
            "http://example.com/page1", 
            headers={'User-Agent': 'TestAgent/1.0'}, 
            timeout=tinyresearch.REQUESTS_TIMEOUT
        )

        # Test already visited URL
        content_pieces_again = tinyresearch.visit_urls(self.context, {"urls": ["http://example.com/page1"]})
        self.assertEqual(len(content_pieces_again), 0) # Should not re-visit

        # Test failed visit (e.g., 403 Forbidden)
        self.context["urls_to_visit"] = ["http://example.com/forbidden"]
        self.context["visited_urls"] = set() # Reset for this sub-test
        mock_response_fail = MagicMock()
        mock_response_fail.raise_for_status.side_effect = tinyresearch.requests.exceptions.HTTPError("403 Forbidden")
        mock_requests_get.return_value = mock_response_fail
        
        content_pieces_fail = tinyresearch.visit_urls(self.context, {"urls": ["http://example.com/forbidden"]})
        self.assertEqual(len(content_pieces_fail), 0)
        self.assertNotIn("http://example.com/forbidden", self.context["visited_urls"]) # Not added if failed

    def test_visit_urls_edge_cases(self):
        # Test with invalid URL format in input
        self.context["urls_to_visit"] = ["not_a_url", "http://example.com/good"]
        self.context["visited_urls"] = set()
        
        def side_effect_for_get(url, headers, timeout):
            if url == "not_a_url":
                raise tinyresearch.requests.exceptions.RequestException("Invalid URL for test")
            elif url == "http://example.com/good":
                mock_resp = MagicMock()
                mock_resp.status_code = 200
                mock_resp.content = b"<p>Good content</p>"
                mock_resp.raise_for_status.return_value = None
                return mock_resp
            raise ValueError(f"Unexpected URL for mock_get in test_visit_urls_edge_cases: {url}")

        with patch('tinyresearch.requests.get', side_effect=side_effect_for_get) as mock_get_call:
            content_pieces = tinyresearch.visit_urls(self.context, {"urls": ["not_a_url", "http://example.com/good"]})
            self.assertEqual(len(content_pieces), 1) # Only good URL should be processed
            self.assertEqual(content_pieces[0]["url"], "http://example.com/good")
            # Check that get was called for the good URL and the bad one (which then failed)
            self.assertIn(unittest.mock.call("http://example.com/good", headers=unittest.mock.ANY, timeout=unittest.mock.ANY), mock_get_call.call_args_list)
            self.assertIn(unittest.mock.call("not_a_url", headers=unittest.mock.ANY, timeout=unittest.mock.ANY), mock_get_call.call_args_list)
            self.assertNotIn("not_a_url", self.context["visited_urls"]) # Ensure it wasn't marked as visited

class TestTinyResearchContextAndFlow(unittest.TestCase):
    @patch('tinyresearch._get_embedding_model') # Patch the embedding model loading
    def test_initialize_context_and_variables(self, mock_get_embedding_model_func):
        # Setup the mock for _get_embedding_model
        mock_model_instance = MagicMock()
        mock_model_instance.get_sentence_embedding_dimension.return_value = 384 # Example dimension for 'all-MiniLM-L6-v2'
        # Mock the encode method to return a 2D numpy array as expected by FAISS
        mock_model_instance.encode.return_value = np.random.rand(1, 384).astype(np.float32)
        mock_get_embedding_model_func.return_value = mock_model_instance

        mock_args = argparse.Namespace(
            token_budget=10000,
            user_question="Test Question?",
            max_simulation_steps=5,
            web_browser_user_agent="TestUserAgent",
            max_url_content_length=1000,
            time_budget_minutes=tinyresearch.DEFAULT_TIME_BUDGET_MINUTES, # Add missing arg
            reasoning_style="LLM Default", # Add missing arg
            reasoning_style_active=False # Add missing arg
        )
        context = tinyresearch.initialize_context_and_variables(mock_args)
        self.assertEqual(context["token_budget"], 10000)
        self.assertEqual(context["user_question"], "Test Question?")
        self.assertEqual(context["max_simulation_steps"], 5)
        self.assertEqual(context["gaps_queue"][0]["text"], "Test Question?")
        self.assertIn("Test Question?", context["known_questions"])
        # _get_embedding_model is called once by _initialize_vector_db and once by add_text_to_vector_db
        self.assertEqual(mock_get_embedding_model_func.call_count, 2) 

    def test_check_token_budget_exceeded(self):
        context_ok = {"tokens_used": 500, "token_budget": 1000}
        self.assertFalse(tinyresearch.check_token_budget_exceeded(context_ok))

        context_exceeded = {"tokens_used": 1000, "token_budget": 1000}
        self.assertTrue(tinyresearch.check_token_budget_exceeded(context_exceeded))

        context_over = {"tokens_used": 1001, "token_budget": 1000}
        self.assertTrue(tinyresearch.check_token_budget_exceeded(context_over))

    @patch('tinyresearch.get_text_response')
    def test_generate_final_answer_with_agent(self, mock_get_text_response):
        # Test with empty knowledge base
        context_empty_kb = {"user_question": "Q?", "knowledge_base": []}
        answer_empty = tinyresearch.generate_final_answer_with_agent(context_empty_kb, "reason", "SummarizerAgent")
        self.assertIn("No significant knowledge gathered", answer_empty)

        # Test with stock data direct answer
        context_stock = {
            "user_question": "NVDA stock price?", 
            "knowledge_base": [{"type": "stock_data", "ticker": "NVDA", "data_text": "NVDA is $100"}]
        }
        answer_stock = tinyresearch.generate_final_answer_with_agent(context_stock, "reason", "SummarizerAgent")
        self.assertEqual(answer_stock, "NVDA is $100")

        # Test with datetime direct answer
        context_time = {
            "user_question": "Time in Rome?", 
            "knowledge_base": [{"type": "current_datetime", "timezone": "Europe/Rome", "datetime": "10:00 Rome"}]
        }
        answer_time = tinyresearch.generate_final_answer_with_agent(context_time, "reason", "SummarizerAgent")
        self.assertEqual(answer_time, "The current date and time in Europe/Rome is 10:00 Rome.") # Added period

        # Test fallback to summarizer LLM call
        context_summarizer = {
            "user_question": "Complex Q?", 
            "knowledge_base": [{"type": "search_result_snippet", "snippet": "Some snippet"}],
            "tokens_used": 0 # Needed for get_text_response
        }
        mock_get_text_response.return_value = "Summarized answer."
        # Mock global LLM settings needed by get_text_response
        with patch.object(tinyresearch, 'g_llm_provider', 'litellm'), \
             patch.object(tinyresearch, 'g_llm_model', 'test/model'), \
             patch.object(tinyresearch, 'g_llm_provider_endpoint', 'http://localhost'):
            answer_summary = tinyresearch.generate_final_answer_with_agent(context_summarizer, "reason", "SummarizerAgent")
        
        self.assertEqual(answer_summary, "Summarized answer.")
        mock_get_text_response.assert_called_once()
        keyword_args = mock_get_text_response.call_args.kwargs # Access keyword arguments
        self.assertEqual(keyword_args['agent_name'], "SummarizerAgent")
        self.assertIn("Complex Q?", keyword_args['user_prompt_text'])

class TestTinyResearchLLMInteractions(unittest.TestCase):

    @patch('tinyresearch._make_llm_api_call')
    def test_decide_json_action_valid_json(self, mock_make_llm_api_call):
        mock_context = {"tokens_used": 0}
        valid_json_response = '{"action_type": "search", "data": {"query": "test query"}}'
        mock_make_llm_api_call.return_value = (valid_json_response, None)

        result = tinyresearch.decide_json_action("TestAgent", "SystemPrompt", "UserPrompt", "litellm", "model", "endpoint", 100, mock_context)
        
        self.assertEqual(result["action_type"], "search")
        self.assertEqual(result["data"]["query"], "test query")
        mock_make_llm_api_call.assert_called_once()

    @patch('tinyresearch._make_llm_api_call')
    def test_decide_json_action_json_in_markdown(self, mock_make_llm_api_call):
        mock_context = {"tokens_used": 0}
        md_json_response = '```json\n{"action_type": "visit", "data": {"urls": ["http://example.com"]}}\n```'
        mock_make_llm_api_call.return_value = (md_json_response, None)

        result = tinyresearch.decide_json_action("TestAgent", "SystemPrompt", "UserPrompt", "litellm", "model", "endpoint", 100, mock_context)
        
        self.assertEqual(result["action_type"], "visit")
        self.assertEqual(result["data"]["urls"], ["http://example.com"])

    @patch('tinyresearch._make_llm_api_call')
    def test_decide_json_action_missing_keys(self, mock_make_llm_api_call):
        mock_context = {"tokens_used": 0}
        invalid_json_response = '{"action_typo": "search", "data": {"query": "test query"}}' # Deliberate typo
        mock_make_llm_api_call.return_value = (invalid_json_response, None)

        result = tinyresearch.decide_json_action("TestAgent", "SystemPrompt", "UserPrompt", "litellm", "model", "endpoint", 100, mock_context)
        
        self.assertEqual(result["action_type"], "error")
        self.assertIn("missing 'action_type' or 'data' keys", result["data"]["message"])

    @patch('tinyresearch._make_llm_api_call')
    def test_decide_json_action_llm_call_error(self, mock_make_llm_api_call):
        mock_context = {"tokens_used": 0}
        mock_make_llm_api_call.return_value = (None, "LLM API Failed")

        result = tinyresearch.decide_json_action("TestAgent", "SystemPrompt", "UserPrompt", "litellm", "model", "endpoint", 100, mock_context)
        
        self.assertEqual(result["action_type"], "error")
        self.assertEqual(result["data"]["message"], "LLM API Failed")

# Example of testing a part of the main loop's action dispatch
class TestRunDeepSearchAgentLogic(unittest.TestCase):
    @patch('tinyresearch.decide_json_action')
    @patch('tinyresearch.execute_get_stock_price') # Mock the actual tool execution
    @patch('tinyresearch.initialize_context_and_variables') # To control initial context
    @patch('tinyresearch.get_current_question_from_gaps')
    @patch('tinyresearch.check_token_budget_exceeded', return_value=False) # Assume budget is fine
    @patch('tinyresearch.time.sleep') # To prevent actual sleep
    def test_run_deep_search_agent_get_stock_price_action(
        self, mock_sleep, mock_check_budget, mock_get_question, 
        mock_init_context, mock_execute_stock, mock_decide_action):

        # Setup mock args for initialize_context_and_variables
        mock_args = argparse.Namespace(
            user_question="NVDA stock price?", token_budget=1000, max_simulation_steps=1,
            web_browser_user_agent="test", max_url_content_length=100,
            llm_provider="litellm", llm_model="test/model", llm_provider_endpoint="http://localhost:11434",
            output_format="text",
            time_budget_minutes=tinyresearch.DEFAULT_TIME_BUDGET_MINUTES, # Add
            reasoning_style="LLM Default", # Add
            reasoning_style_active=False # Add
        )
        # Mock global LLM settings that run_deep_search_agent sets up
        tinyresearch.g_llm_provider = mock_args.llm_provider
        tinyresearch.g_llm_model = mock_args.llm_model
        # Setup initial context (simplified for this test)
        initial_context = {
            "simulation_step": 0, "max_simulation_steps": 1, 
            "gaps_queue": [], "knowledge_base": [], 
            "urls_to_visit": [], "visited_urls": set(), 
            "bad_attempts": [], "known_questions": set(), 
            "tokens_used": 0, "token_budget": 1000, 
            "user_question": "NVDA stock price?",
            "current_question": None, 
            "start_time": time.time(), # Add
            "time_budget_seconds": tinyresearch.DEFAULT_TIME_BUDGET_MINUTES * 60, # Add
            "is_user_question_context": False,
            "disable_reflect_next": False, "disable_search_next": False, "disable_visit_next": False
        }
        mock_init_context.return_value = initial_context
        mock_get_question.return_value = {"text": "NVDA stock price?", "is_original": True} # Simulate a question being popped

        # Simulate MainDecisionAgent choosing 'get_stock_price'
        mock_decide_action.return_value = {"action_type": "get_stock_price", "data": {"ticker": "NVDA"}}

        tinyresearch.run_deep_search_agent(mock_args)

        mock_execute_stock.assert_called_once_with(initial_context, {"ticker": "NVDA"})

    @patch('tinyresearch.decide_json_action')
    @patch('tinyresearch.execute_search')
    @patch('tinyresearch.execute_wikipedia_search')
    @patch('tinyresearch.execute_arxiv_search')
    @patch('tinyresearch.initialize_context_and_variables')
    @patch('tinyresearch.get_current_question_from_gaps')
    @patch('tinyresearch.check_token_budget_exceeded', return_value=False)
    @patch('tinyresearch.time.sleep')
    def test_run_deep_search_agent_search_dispatch(
        self, mock_sleep, mock_check_budget, mock_get_question, mock_init_context,
        mock_exec_arxiv, mock_exec_wiki, mock_exec_search, mock_decide_action_stack):

        mock_args = argparse.Namespace(
            user_question="Search for X", token_budget=1000, max_simulation_steps=1,
            web_browser_user_agent="test", max_url_content_length=100,
            llm_provider="litellm", llm_model="test/model", llm_provider_endpoint="http://localhost:11434",
            output_format="text",
            time_budget_minutes=tinyresearch.DEFAULT_TIME_BUDGET_MINUTES, # Add
            reasoning_style="LLM Default", # Add
            reasoning_style_active=False # Add
        )
        tinyresearch.g_llm_provider = mock_args.llm_provider
        tinyresearch.g_llm_model = mock_args.llm_model

        initial_context = {
            "simulation_step": 0, "max_simulation_steps": 1, "gaps_queue": [], "knowledge_base": [],
            "urls_to_visit": [], "visited_urls": set(), "bad_attempts": [], "known_questions": set(),
            "tokens_used": 0, "token_budget": 1000, "user_question": "Search for X",
            "current_question": None, 
            "start_time": time.time(), # Add
            "time_budget_seconds": tinyresearch.DEFAULT_TIME_BUDGET_MINUTES * 60, # Add
            "is_user_question_context": False,
            "disable_reflect_next": False, "disable_search_next": False, "disable_visit_next": False
        }
        mock_init_context.return_value = initial_context
        mock_get_question.return_value = {"text": "Search for X", "is_original": True}

        # Test dispatch to Wikipedia
        mock_decide_action_stack.side_effect = [
            {"action_type": "search", "data": {"query": "X"}}, # MainDecisionAgent
            {"action_type": "search_strategy_determined", "data": {"engine_to_use": "wikipedia", "refined_query": "X wiki"}} # SearchStrategyAgent
        ]
        tinyresearch.run_deep_search_agent(mock_args)
        mock_exec_wiki.assert_called_once_with(initial_context, "X wiki")
        mock_exec_search.assert_not_called()
        mock_exec_arxiv.assert_not_called()

        # Reset mocks for next scenario
        mock_exec_wiki.reset_mock()
        mock_decide_action_stack.reset_mock() # Important to reset side_effect counter
        initial_context["simulation_step"] = 0 # Reset step for the next run

        # Test dispatch to Arxiv
        mock_decide_action_stack.side_effect = [
            {"action_type": "search", "data": {"query": "Y"}}, # MainDecisionAgent
            {"action_type": "search_strategy_determined", "data": {"engine_to_use": "arxiv", "refined_query": "Y arxiv"}} # SearchStrategyAgent
        ]
        tinyresearch.run_deep_search_agent(mock_args) # Call again with new side_effect
        mock_exec_arxiv.assert_called_once_with(initial_context, "Y arxiv")
        mock_exec_search.assert_not_called()
        mock_exec_wiki.assert_not_called()

    @patch('tinyresearch.decide_json_action')
    @patch('tinyresearch.evaluate_answer_with_agent')
    @patch('tinyresearch.store_bad_attempt_reset_context')
    @patch('tinyresearch.initialize_context_and_variables')
    @patch('tinyresearch.get_current_question_from_gaps')
    @patch('tinyresearch.check_token_budget_exceeded', return_value=False)
    @patch('tinyresearch.time.sleep')
    def test_run_deep_search_agent_answer_action_flow(
        self, mock_sleep, mock_check_budget, mock_get_question, mock_init_context,
        mock_store_bad_attempt, mock_eval_answer, mock_decide_action):

        mock_args = argparse.Namespace(
            user_question="Q?", token_budget=1000, max_simulation_steps=1,
            web_browser_user_agent="test", max_url_content_length=100,
            llm_provider="litellm", llm_model="test/model", llm_provider_endpoint="http://localhost:11434",
            output_format="text",
            time_budget_minutes=tinyresearch.DEFAULT_TIME_BUDGET_MINUTES, # Add
            reasoning_style="LLM Default", # Add
            reasoning_style_active=False # Add
        )
        tinyresearch.g_llm_provider = mock_args.llm_provider
        tinyresearch.g_llm_model = mock_args.llm_model

        initial_context = {
            "simulation_step": 0, "max_simulation_steps": 1, "gaps_queue": [], "knowledge_base": [],
            "urls_to_visit": [], "visited_urls": set(), "bad_attempts": [], "known_questions": set(),
            "tokens_used": 0, "token_budget": 1000, "user_question": "Q?",
            "current_question": "Q?", 
            "start_time": time.time(), # Add
            "time_budget_seconds": tinyresearch.DEFAULT_TIME_BUDGET_MINUTES * 60, # Add
            "is_user_question_context": True, # Assume it's the original question
            "disable_reflect_next": False, "disable_search_next": False, "disable_visit_next": False
        }
        mock_init_context.return_value = initial_context
        mock_get_question.return_value = {"text": "Q?", "is_original": True}

        # Scenario 1: Good, definitive answer
        mock_decide_action.return_value = {"action_type": "answer", "data": {"text": "Final Answer", "is_definitive": True, "is_for_original": True, "has_references": True}}
        mock_eval_answer.return_value = True # Evaluator says it's good
        
        result = tinyresearch.run_deep_search_agent(mock_args)
        # In this setup, run_deep_search_agent doesn't directly return the final_result,
        # but we can infer by checking if store_bad_attempt was NOT called and eval_answer was.
        # A more direct way would be to check a global or a passed-in mutable object if run_deep_search_agent modified it.
        # For this test, we'll check that the flow proceeded as if a good answer was found.
        mock_eval_answer.assert_called_once()
        mock_store_bad_attempt.assert_not_called()

        # Scenario 2: Bad answer
        mock_eval_answer.reset_mock()
        mock_store_bad_attempt.reset_mock()
        mock_decide_action.return_value = {"action_type": "answer", "data": {"text": "Bad Answer", "is_definitive": False, "is_for_original": True}}
        mock_eval_answer.return_value = False # Evaluator says it's bad
        
        tinyresearch.run_deep_search_agent(mock_args)
        mock_eval_answer.assert_called_once()
        mock_store_bad_attempt.assert_called_once()

    @patch('tinyresearch.decide_json_action')
    @patch('tinyresearch.store_bad_attempt_reset_context')
    @patch('tinyresearch.initialize_context_and_variables')
    @patch('tinyresearch.get_current_question_from_gaps')
    @patch('tinyresearch.check_token_budget_exceeded', return_value=False)
    @patch('tinyresearch.time.sleep')
    def test_run_deep_search_agent_error_action(
        self, mock_sleep, mock_check_budget, mock_get_question, mock_init_context,
        mock_store_bad_attempt, mock_decide_action):

        mock_args = argparse.Namespace(
            user_question="Q?", token_budget=1000, max_simulation_steps=1,
            web_browser_user_agent="test", max_url_content_length=100,
            llm_provider="litellm", llm_model="test/model", llm_provider_endpoint="http://localhost:11434",
            output_format="text",
            time_budget_minutes=tinyresearch.DEFAULT_TIME_BUDGET_MINUTES, # Add
            reasoning_style="LLM Default", # Add
            reasoning_style_active=False # Add
        )
        tinyresearch.g_llm_provider = mock_args.llm_provider
        tinyresearch.g_llm_model = mock_args.llm_model

        initial_context = {
            "simulation_step": 0, "max_simulation_steps": 1, "gaps_queue": [], "knowledge_base": [],
            "urls_to_visit": [], "visited_urls": set(), "bad_attempts": [], "known_questions": set(),
            "tokens_used": 0, "token_budget": 1000, "user_question": "Q?",
            "current_question": "Q?", 
            "start_time": time.time(), # Add
            "time_budget_seconds": tinyresearch.DEFAULT_TIME_BUDGET_MINUTES * 60, # Add
            "is_user_question_context": True,
            "disable_reflect_next": False, "disable_search_next": False, "disable_visit_next": False
        }
        mock_init_context.return_value = initial_context
        mock_get_question.return_value = {"text": "Q?", "is_original": True}

        mock_decide_action.return_value = {"action_type": "error", "data": {"message": "LLM Error", "agent_name": "MainDecisionAgent"}}
        
        tinyresearch.run_deep_search_agent(mock_args)
        mock_store_bad_attempt.assert_called_once()


if __name__ == '__main__':
    unittest.main()