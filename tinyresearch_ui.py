import streamlit as st
import sys
import os
import argparse
import io
import contextlib
import json # Import the json module
import litellm
import time # For simulating streaming if needed, and for unique keys

# Add current directory to Python path to import tinyresearch
# This assumes tinyresearch_ui.py is in the same directory as tinyresearch.py
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Torch is not a direct dependency for core logic anymore.
# If other libraries (like faiss) pull it in, their own threading management applies.

try:
    import tinyresearch
except ImportError as e:
    st.error(f"Failed to import tinyresearch.py. Ensure it's in the same directory or Python path. Error: {e}")
    st.stop()

# Versioning scheme: YYYY.MM.DD.progressive_letter (e.g., 2024.05.10.a)
__version__ = "2024.05.10.c"

# --- LiteLLM Debugging ---
# os.environ['LITELLM_LOG'] = 'DEBUG' # Enable verbose logging for debugging

# --- App Configuration ---
APP_TITLE = "TinyResearch UI"
APP_VERSION = __version__ # Use the UI's own version string
GITHUB_LINK = "https://github.com/PieBru/TinyResearch" # Please verify/update this link
APP_DESCRIPTION = "An AI agent for answering complex questions by breaking them down, searching, and synthesizing information."
SETTINGS_FILE = "tinyresearch_ui_settings.json"

# --- Callback Functions (Define these early) ---
def update_llm_config_from_ui():
    """Callback to update LLM config from UI controls."""
    # Read the value from the selectbox's specific key
    st.session_state.selected_style = st.session_state.selected_style_selectbox
    # Read the value from the toggle's specific key
    st.session_state.style_active = st.session_state.style_active_toggle
    
    # Apply preset if style is active and not "LLM Default"
    if st.session_state.style_active and st.session_state.selected_style != "LLM Default":
        preset = REASONING_STYLES.get(st.session_state.selected_style)
        if preset:
            st.session_state.llm_provider = preset["provider"]
            st.session_state.llm_endpoint = preset["endpoint"]
            st.session_state.llm_model = preset["model"]
            st.session_state.llm_temperature = preset["temperature"]
            st.session_state.llm_top_p = preset["top_p"]
            st.session_state.llm_top_k = preset["top_k"]
            st.session_state.llm_repeat_penalty = preset["repeat_penalty"]
            st.session_state.llm_min_p = preset["min_p"]
    # This callback is just to keep selected_style and style_active in sync
    # and to trigger a rerun so the UI (e.g. toggle's disabled state) updates.

def auto_update_model_on_provider_change():
    """Callback to auto-update model name when provider changes."""
    provider = st.session_state.get("llm_provider_selectbox_manual", st.session_state.llm_provider)
    
    if provider == "gemini":
        st.session_state.llm_model = "gemini-2.0-flash"
    elif provider == "litellm":
        if st.session_state.llm_model is None or st.session_state.llm_model.startswith("gemini-") or st.session_state.llm_model.startswith("models/gemini-"):
            st.session_state.llm_model = tinyresearch.DEFAULT_LLM_MODEL
    st.session_state.llm_provider = provider
    save_settings()

# --- Settings Persistence ---
def save_settings():
    settings_to_save = {
        'llm_provider': st.session_state.llm_provider,
        'llm_endpoint': st.session_state.llm_endpoint,
        'llm_model': st.session_state.llm_model,
        'llm_temperature': st.session_state.llm_temperature,
        'llm_top_p': st.session_state.llm_top_p,
        'llm_top_k': st.session_state.llm_top_k,
        'llm_repeat_penalty': st.session_state.llm_repeat_penalty,
        'llm_min_p': st.session_state.llm_min_p,
        'selected_style_selectbox': st.session_state.selected_style_selectbox, # Save the selectbox value
        'style_active_toggle': st.session_state.style_active_toggle, # Save the toggle value
        'time_budget': st.session_state.get('time_budget', tinyresearch.DEFAULT_TIME_BUDGET_MINUTES),
        'token_budget': st.session_state.get('token_budget', tinyresearch.DEFAULT_TOKEN_BUDGET),
        'max_steps': st.session_state.get('max_steps', tinyresearch.DEFAULT_MAX_SIMULATION_STEPS),
        'user_agent': st.session_state.get('user_agent', tinyresearch.DEFAULT_WEB_BROWSER_USER_AGENT),
        'max_url_len': st.session_state.get('max_url_len', tinyresearch.DEFAULT_MAX_URL_CONTENT_LENGTH),
    }
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(settings_to_save, f, indent=2)
        print(f"UI Settings saved to {SETTINGS_FILE}") # For CLI log
    except Exception as e:
        print(f"Error saving UI settings: {e}") # For CLI log
        st.sidebar.error(f"Error saving settings: {e}")

def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                loaded_settings = json.load(f)
                st.session_state.update(loaded_settings) # Update session state with loaded settings
                print(f"UI Settings loaded from {SETTINGS_FILE}") # For CLI log
                # Ensure dependent states are also set if needed after loading
                # For example, if selected_style_selectbox is loaded, selected_style should match
                if 'selected_style_selectbox' in loaded_settings:
                    st.session_state.selected_style = loaded_settings['selected_style_selectbox']
                if 'style_active_toggle' in loaded_settings:
                     st.session_state.style_active = loaded_settings['style_active_toggle']
                # Trigger callbacks if necessary after loading, or ensure UI reflects loaded state
                # These functions must be defined before load_settings is called.
                auto_update_model_on_provider_change() # To set model and endpoint disabled status
                update_llm_config_from_ui() # To sync style and active status
                # if 'ui_db_initialized_once' not in st.session_state:
                #     initialize_shared_db() # Call the new DB init function
                #     st.session_state.ui_db_initialized_once = True

    except Exception as e:
        print(f"Error loading UI settings: {e}") # For CLI log
        st.sidebar.warning(f"Could not load previous settings: {e}")

# --- Helper to capture stdout ---
@contextlib.contextmanager
def st_capture_stdout_to_placeholder(placeholder):
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    buffer = ""
    try:
        yield
    finally:
        buffer = captured_output.getvalue()
        placeholder.text_area("Thinking Log:", value=buffer, height=300, key=f"thinking_log_{time.time()}")
        sys.stdout = old_stdout


# --- Main UI ---
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Initialize session state variables
if 'thinking_process_log' not in st.session_state:
    st.session_state.thinking_process_log = ""
if 'final_answer_output' not in st.session_state:
    st.session_state.final_answer_output = ""
if 'is_running' not in st.session_state:
    st.session_state.is_running = False
if 'direct_chat_history' not in st.session_state:
    st.session_state.direct_chat_history = [] # For "Ask LLM (Direct)"
if 'ui_tokens_used' not in st.session_state: # For tracking tokens used by UI DB operations
    st.session_state.ui_tokens_used = 0

# Initialize session state with defaults, then try to load saved settings
def initialize_session_state_with_defaults():
    defaults = {
        'llm_provider': tinyresearch.DEFAULT_LLM_PROVIDER,
        'llm_endpoint': tinyresearch.DEFAULT_LLM_PROVIDER_ENDPOINT,
        'llm_model': tinyresearch.DEFAULT_LLM_MODEL,
        'llm_temperature': tinyresearch.LLM_TEMPERATURE,
        'llm_top_p': tinyresearch.LLM_TOP_P,
        'llm_top_k': tinyresearch.LLM_TOP_K,
        'llm_repeat_penalty': tinyresearch.LLM_REPEAT_PENALTY,
        'llm_min_p': tinyresearch.LLM_MIN_P,
        'selected_style': "LLM Default",
        'selected_style_selectbox': "LLM Default",
        'style_active': False,
        'style_active_toggle': False, # Add toggle specific state
        'time_budget': tinyresearch.DEFAULT_TIME_BUDGET_MINUTES,
        'token_budget': tinyresearch.DEFAULT_TOKEN_BUDGET,
        'max_steps': tinyresearch.DEFAULT_MAX_SIMULATION_STEPS,
        'user_agent': tinyresearch.DEFAULT_WEB_BROWSER_USER_AGENT,
        'max_url_len': tinyresearch.DEFAULT_MAX_URL_CONTENT_LENGTH,
        'embedding_model_name': tinyresearch.DEFAULT_LITELLM_EMBEDDING_MODEL, # Add embedding model
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def initialize_shared_db():
    """Initializes the shared vector DB components in session_state if not already done."""
    if 'vector_db_index' not in st.session_state or st.session_state.vector_db_index is None:
        st.session_state.vector_db_index = None # Explicitly set to None before init
        st.session_state.vector_db_texts = []   # Explicitly set to empty list
        
        # Prepare a context dict for the initial DB setup
        # This context will be modified in-place by _initialize_vector_db
        initial_db_context = {
            "vector_db_index": st.session_state.vector_db_index,
            "vector_db_texts": st.session_state.vector_db_texts,
            "embedding_model_name": st.session_state.embedding_model_name,
            "tokens_used": st.session_state.ui_tokens_used # Use UI specific token counter
        }
        print("UI: Initializing shared vector DB...")
        try:
            tinyresearch._initialize_vector_db(initial_db_context) # This can raise ValueError
            # Update session state with the initialized DB components
            st.session_state.vector_db_index = initial_db_context["vector_db_index"]
            st.session_state.vector_db_texts = initial_db_context["vector_db_texts"]
            st.session_state.embedding_dim = initial_db_context.get("embedding_dim") 
            st.session_state.ui_tokens_used = initial_db_context["tokens_used"] # Update token count
            print("UI: Shared vector DB initialized successfully.")
        except ValueError as e: # Specifically catch the dimension error
            st.session_state.embedding_dim = None # Mark as not determined
            # Store the error to display it, and allow retrying
            st.session_state.db_init_error = f"Could not initialize shared Vector DB on startup. Embedding dimension for '{st.session_state.embedding_model_name}' could not be determined. Error: {e}. Ensure Ollama is running with the model and try again."
            # st.error(f"UI Warning: {st.session_state.db_init_error}") # Displayed below
            print(f"UI Warning: Failed to initialize shared vector DB: {e}")


if 'settings_loaded_once' not in st.session_state:
    initialize_session_state_with_defaults()
    load_settings()
    if 'ui_db_initialized_once' not in st.session_state: # If load_settings didn't trigger it
        initialize_shared_db()
        st.session_state.ui_db_initialized_once = True
    st.session_state.settings_loaded_once = True # Ensure settings are loaded only once per session start

# --- LLM Presets Definition ---
OLLAMA_ENDPOINT = "http://localhost:11434" # Default Ollama endpoint
REASONING_STYLES = {
    "LLM Default": {},
    "Qwen3 (Ollama)": {"provider": "litellm", "endpoint": OLLAMA_ENDPOINT, "model": "ollama/qwen2.5", "temperature": 0.1, "top_p": 0.1, "top_k": 40, "repeat_penalty": 1.05, "min_p": 0.0},
    "DeepHermes3 (Ollama)": {"provider": "litellm", "endpoint": OLLAMA_ENDPOINT, "model": "ollama/DeepHermes3", "temperature": 0.2, "top_p": 0.2, "top_k": 50, "repeat_penalty": 1.1, "min_p": 0.0},
 }
STYLE_NAMES = list(REASONING_STYLES.keys())

def get_shared_db_context_for_ui():
    """Helper to get the current shared DB context from session_state for UI operations."""
    return {
        "vector_db_index": st.session_state.vector_db_index,
        "vector_db_texts": st.session_state.vector_db_texts,
        "embedding_model_name": st.session_state.embedding_model_name,
        "embedding_dim": st.session_state.get("embedding_dim"), # Pass pre-calculated dim
        "tokens_used": st.session_state.ui_tokens_used 
    }


# --- Sidebar ---
with st.sidebar: # This line was causing the IndentationError due to the elif above it lacking a body
    st.subheader(APP_TITLE) # Added app title to the sidebar
    with st.expander("LLM Configuration", expanded=False):
        # The selectbox updates its own key, which the callback then reads.
        # The general st.session_state.llm_provider is updated by the callback.
        # The key "llm_provider_selectbox_manual" will hold the current selection of this widget.
        selected_provider_for_ui = st.selectbox("Provider", options=["gemini", "litellm"],
                                     index=["gemini", "litellm"].index(st.session_state.llm_provider), 
                                     key="llm_provider_selectbox_manual",
                                     on_change=auto_update_model_on_provider_change,
                                     help="Select the LLM provider. Settings will be saved.")
        
        # Determine if endpoint should be disabled based on the CURRENT selection of the provider selectbox
        # The 'selected_provider_for_ui' variable holds the current value from the selectbox above.
        endpoint_is_disabled_now = (selected_provider_for_ui == "gemini")

        st.session_state.llm_endpoint = st.text_input("Endpoint", value=st.session_state.llm_endpoint,
                                      help="e.g., http://localhost:11434 for local Ollama. Disabled for Gemini.", key="llm_endpoint_input_manual", disabled=endpoint_is_disabled_now)
                                      # Removed on_change=save_settings from individual inputs, will save on action
        st.session_state.llm_model = st.text_input("Model Name", value=st.session_state.llm_model,
                                   help="e.g., ollama/qwen2.5 or models/gemini-1.5-flash-latest", key="llm_model_input_manual")
        

        # Temperature Slider
        st.session_state.llm_temperature = st.slider(
            "Temperature (Creativity)", 
            min_value=0.0, max_value=2.0, 
            value=st.session_state.llm_temperature, 
            step=0.01, key="llm_temperature_slider",
            help="Controls randomness. Lower is more deterministic.",
            on_change=save_settings
        )

        # Top_P Slider
        st.session_state.llm_top_p = st.slider(
            "Top P (Nucleus Sampling)",
            min_value=0.0, max_value=1.0,
            value=st.session_state.llm_top_p,
            step=0.01, key="llm_top_p_slider",
            help="Filters vocabulary to the smallest set whose cumulative probability exceeds P. 0.1 means 10%.",
            on_change=save_settings
        )

        # Top_K Slider
        st.session_state.llm_top_k = st.slider(
            "Top K",
            min_value=0, max_value=100, # 0 often means disabled
            value=st.session_state.llm_top_k,
            step=1, key="llm_top_k_slider",
            help="Limits the next token to one of the top-k most probable tokens. (0 to disable)",
            on_change=save_settings
        )

        # Repeat Penalty Slider
        st.session_state.llm_repeat_penalty = st.slider(
            "Repeat Penalty",
            min_value=1.0, max_value=2.0,
            value=st.session_state.llm_repeat_penalty,
            step=0.01, key="llm_repeat_penalty_slider",
            help="How much to discourage repeating the same token (e.g., frequency penalty). 1.0 = no penalty.",
            on_change=save_settings
        )

        # Min_P Slider
        st.session_state.llm_min_p = st.slider(
            "Min P",
            min_value=0.0, max_value=0.1, # Typically a small value
            value=st.session_state.llm_min_p,
            step=0.001, key="llm_min_p_slider",
            help="Minimum base probability for a token to be selected. (Support varies by model)",
            on_change=save_settings
        )
        st.selectbox(
            "LLM Thinking/Reasoning Preset", STYLE_NAMES,
            # Use the selectbox's own state key for its index/default value
            index=STYLE_NAMES.index(st.session_state.get("selected_style_selectbox", "LLM Default")), 
            key="selected_style_selectbox", # Key for this widget
            on_change=lambda: (update_llm_config_from_ui(), save_settings()) # Combined on_change
        )

        st.toggle(
            "Activate Selected Style (uses Ollama)",
            value=st.session_state.get("style_active_toggle", False) if st.session_state.get("selected_style_selectbox", "LLM Default") != "LLM Default" else False,
            key="style_active_toggle", # Key for this widget, its value is read in update_llm_config_from_ui
            disabled=(st.session_state.selected_style_selectbox == "LLM Default"), # Check the selectbox's current value
            on_change=lambda: (update_llm_config_from_ui(), save_settings()), # Combined on_change
            help="ON = Slower, but best for tasks requiring detailed reasoning, such as solving mathematical problems, coding, or logical deductions. It enhances accuracy by allowing the model to \"think\" through the problem before responding."
        )

    with st.expander("Research Options", expanded=False):
        # Use session state keys directly for these inputs and add on_change to save settings
        st.session_state.time_budget = st.number_input("Time Budget (minutes)", min_value=1, 
                                                       value=st.session_state.time_budget, step=1, 
                                                       key="time_budget_input", on_change=save_settings)
        st.session_state.token_budget = st.number_input("Token Budget", min_value=1000, 
                                                        value=st.session_state.token_budget, step=1000, 
                                                        key="token_budget_input", on_change=save_settings)
        st.session_state.max_steps = st.number_input("Max Simulation Steps", min_value=1, 
                                                     value=st.session_state.max_steps, step=1, 
                                                     key="max_steps_input", on_change=save_settings)
        st.session_state.user_agent = st.text_input("Web Browser User Agent", 
                                                    value=st.session_state.user_agent, 
                                                    key="user_agent_input", on_change=save_settings)
        st.session_state.max_url_len = st.number_input("Max URL Content Length", min_value=100, 
                                                       value=st.session_state.max_url_len, step=100, 
                                                       key="max_url_len_input", on_change=save_settings)

        st.session_state.embedding_model_name = st.text_input("Embedding Model Name", 
                                                              value=st.session_state.embedding_model_name, 
                                                              key="embedding_model_name_input", on_change=save_settings)

    with st.expander("About TinyResearch", expanded=False):
        st.write(f"**Version:** {APP_VERSION}")
        st.write(APP_DESCRIPTION)
        st.markdown(f"[GitHub Repository]({GITHUB_LINK})")

# --- Main Content Area ---
user_question_ui = st.text_input("Enter your question:", key="user_question_input", disabled=st.session_state.is_running)

col1, col2 = st.columns(2)

def run_direct_llm_interaction():
    st.session_state.is_running = True
    st.session_state.thinking_process_log = "Calling LLM directly...\n"
    st.session_state.final_answer_output = "" # Clear previous answer
    save_settings() # Save settings before running
    
    # Determine system prompt and modify user question based on style and toggle state
    current_user_query = user_question_ui
    system_prompt_for_direct_chat = "You are a helpful AI assistant. Provide a concise and direct answer." # Default

    selected_style_ui = st.session_state.get("selected_style_selectbox", "LLM Default")

    if selected_style_ui == "Qwen3":
        if st.session_state.style_active_toggle:
            processed_user_question += " /think"
        else: # Qwen3 selected, but toggle is OFF
            # For direct chat, we might not append /no_think unless it's specifically handled by the model
            pass 
    elif selected_style_ui == "DeepHermes3":
        if st.session_state.style_active_toggle:
            system_prompt_for_direct_chat = tinyresearch.DEEPHERMES3_SYSTEM_PROMPT 

    # Add user query to shared vector DB
    db_context = get_shared_db_context_for_ui()
    tinyresearch.add_text_to_vector_db(db_context, current_user_query, source_info="direct_user_query")
    st.session_state.ui_tokens_used = db_context["tokens_used"] # Update token count
    # Note: vector_db_index and vector_db_texts are modified in-place in db_context,
    # and since db_context gets its values from st.session_state, they are effectively updated.

    # Add current user query to direct chat history for this session
    # The user's query is added to history here, assistant's response is added after it's received.
    st.session_state.direct_chat_history.append({"role": "user", "content": current_user_query})
    print(f"UI_DIRECT_CHAT: Added user query to history. Current embedding model for DB: '{st.session_state.embedding_model_name}'") # DEBUG

    # Construct messages for LLM call
    messages_for_llm = [{"role": "system", "content": system_prompt_for_direct_chat}]
    # Add some recent history from direct_chat_history (e.g., last 5 turns, 10 messages)
    # For simplicity, let's take last N messages.
    MAX_HISTORY_MESSAGES = 10 
    messages_for_llm.extend(st.session_state.direct_chat_history[-MAX_HISTORY_MESSAGES:])

    st.session_state.thinking_process_log += f"--- Input Messages to LLM ---\n{json.dumps(messages_for_llm, indent=2)}\n---------------------------\n"

    start_time_op = time.time() # Record start time
    temp_context = {"tokens_used": 0}
    
    # Use effective LLM config from session state
    current_llm_provider = st.session_state.llm_provider
    current_llm_model = st.session_state.llm_model
    current_llm_endpoint = st.session_state.llm_endpoint

    if current_llm_provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            st.error("GEMINI_API_KEY environment variable not set for 'gemini' provider!")
            st.session_state.is_running = False
            return
        try:
            tinyresearch.genai.configure(api_key=api_key)
            tinyresearch.g_gemini_api_key = api_key
        except Exception as e:
            st.error(f"Error configuring Gemini: {e}")
            st.session_state.is_running = False
            return

    tinyresearch.g_llm_provider = current_llm_provider
    tinyresearch.g_llm_model = current_llm_model
    tinyresearch.g_llm_provider_endpoint = current_llm_endpoint
    tinyresearch.LLM_TEMPERATURE = st.session_state.llm_temperature # Update global temperature
    tinyresearch.LLM_TOP_P = st.session_state.llm_top_p # Update global Top P
    tinyresearch.LLM_TOP_K = st.session_state.llm_top_k
    tinyresearch.LLM_REPEAT_PENALTY = st.session_state.llm_repeat_penalty
    tinyresearch.LLM_MIN_P = st.session_state.llm_min_p

    # --- CLI DEBUG PRINTS ---
    # print(f"UI_DEBUG: Preparing to call tinyresearch.get_text_response from run_direct_llm_interaction")
    # print(f"UI_DEBUG: agent_name='DirectLLMUI'")
    # print(f"UI_DEBUG: system_prompt='{system_prompt_to_use}'") # Be mindful if this is very long
    # print(f"UI_DEBUG: llm_provider='{current_llm_provider}'")
    # print(f"UI_DEBUG: llm_model='{current_llm_model}'")
    # print(f"UI_DEBUG: llm_provider_endpoint='{current_llm_endpoint}'") # THIS IS KEY
    # --- END CLI DEBUG PRINTS ---

    log_capture = io.StringIO()
    try:
        with contextlib.redirect_stdout(log_capture):
            llm_response_text = tinyresearch.get_text_response(
                agent_name="DirectLLMUI", 
                messages=messages_for_llm, # Pass the constructed messages list
                llm_provider=current_llm_provider,
                llm_model=current_llm_model, llm_provider_endpoint=current_llm_endpoint,
                max_output_tokens=tinyresearch.LLM_MAX_OUTPUT_TOKENS, context_for_token_counting=temp_context
            )
        st.session_state.thinking_process_log += log_capture.getvalue()
        st.session_state.final_answer_output = llm_response_text # This is just for display
        duration_op = time.time() - start_time_op
        # Add LLM response to direct chat history and shared vector DB
        st.session_state.direct_chat_history.append({"role": "assistant", "content": llm_response_text})
        db_context_after_response = get_shared_db_context_for_ui()
        print(f"UI_DIRECT_CHAT: Adding to DB with embedding model '{st.session_state.embedding_model_name}': '{llm_response_text[:50]}...'") # DEBUG
        tinyresearch.add_text_to_vector_db(db_context_after_response, llm_response_text, source_info="direct_llm_response")
        st.session_state.ui_tokens_used = db_context_after_response["tokens_used"]
        st.success(f"LLM response received after {duration_op:.1f} seconds.")
    except Exception as e:
        st.session_state.thinking_process_log += log_capture.getvalue()
        st.session_state.thinking_process_log += f"\nEXCEPTION during Direct LLM call: {str(e)}"
        st.error(f"An error occurred with the LLM: {str(e)}")
    finally:
        st.session_state.is_running = False
        # st.rerun() # Rerun to update UI state correctly after operation

def run_tiny_researcher_agent():
    st.session_state.is_running = True
    st.session_state.thinking_process_log = "Starting Tiny Researcher...\n"
    save_settings() # Save settings before running
    st.session_state.final_answer_output = ""

    # Modify user question based on Qwen3 style selection and toggle state
    processed_user_question_for_agent = user_question_ui # Start with the original question
    selected_style_for_agent = st.session_state.get("selected_style_selectbox", "LLM Default")
    style_active_for_agent = st.session_state.style_active_toggle

    if selected_style_for_agent == "Qwen3":
        if st.session_state.style_active_toggle:
            processed_user_question_for_agent += " /think"
        else: # Qwen3 selected, but toggle is OFF
            processed_user_question_for_agent += " /no_think"
    args = argparse.Namespace(
        user_question=processed_user_question_for_agent, 
        token_budget=st.session_state.token_budget, # Use session state
        max_simulation_steps=st.session_state.max_steps, # Use session state
        web_browser_user_agent=st.session_state.user_agent, # Use session state
        time_budget_minutes=st.session_state.time_budget, # Use session state
        shared_vector_db_index=st.session_state.vector_db_index, # Pass shared DB
        shared_vector_db_texts=st.session_state.vector_db_texts,   # Pass shared DB
        shared_embedding_dim=st.session_state.embedding_dim,     # Pass shared DB
        embedding_model_name=st.session_state.embedding_model_name, # Pass embedding model
        max_url_content_length=st.session_state.max_url_len, # Use session state
        llm_provider=st.session_state.llm_provider, # Use effective config
        llm_provider_endpoint=st.session_state.llm_endpoint, # Use effective config
        llm_model=st.session_state.llm_model, # Use effective config
        output_format="text",
        reasoning_style=selected_style_for_agent,
        reasoning_style_active=style_active_for_agent
    )

    # Ensure the global temperature in tinyresearch module is set from UI for this run
    tinyresearch.LLM_TEMPERATURE = st.session_state.llm_temperature
    tinyresearch.LLM_TOP_P = st.session_state.llm_top_p # Update global Top P
    tinyresearch.LLM_TOP_K = st.session_state.llm_top_k
    tinyresearch.LLM_REPEAT_PENALTY = st.session_state.llm_repeat_penalty
    tinyresearch.LLM_MIN_P = st.session_state.llm_min_p

    log_capture = io.StringIO()
    final_answer_researcher = "Error during research."
    start_time_researcher = time.time() # Record start time for this specific run
    try:
        with contextlib.redirect_stdout(log_capture):
            final_answer_researcher = tinyresearch.run_deep_search_agent(args)
        st.session_state.thinking_process_log += log_capture.getvalue()
        st.session_state.final_answer_output = final_answer_researcher
        duration_op = time.time() - start_time_researcher
        st.success(f"Tiny Researcher finished after {duration_op:.1f} seconds.")
    except SystemExit as e:
        st.session_state.thinking_process_log += log_capture.getvalue()
        st.session_state.thinking_process_log += f"\nERROR: Tiny Researcher exited (Code: {e.code}). This might be due to a missing API key or other critical error."
        st.error(f"Tiny Researcher exited. Check thinking process. (Code: {e.code})")
    except Exception as e:
        duration_op = time.time() - start_time_researcher
        st.session_state.thinking_process_log += log_capture.getvalue()
        st.session_state.thinking_process_log += f"\nEXCEPTION during Tiny Researcher run: {str(e)}"
        st.error(f"An error occurred: {str(e)} (Operation took {duration_op:.2f} seconds before error)")
    finally:
        st.session_state.is_running = False
        # st.rerun() # Rerun to update UI state correctly after operation

with col1:
    if st.button("💬 Ask LLM (Direct)", key="ask_llm_button", on_click=run_direct_llm_interaction, disabled=st.session_state.is_running or not user_question_ui):
        pass 

with col2:
    if st.button("🧠 Ask Tiny Researcher", key="ask_tiny_researcher_button", on_click=run_tiny_researcher_agent, disabled=st.session_state.is_running or not user_question_ui):
        pass 

if st.session_state.is_running:
    st.info("Processing your request, please wait...")

# Display DB Initialization Error and Retry Button if applicable
if 'db_init_error' in st.session_state and st.session_state.db_init_error:
    st.warning(st.session_state.db_init_error)
    if st.button("🔄 Retry Vector DB Initialization"):
        st.session_state.db_init_error = None # Clear error
        initialize_shared_db() # Attempt to re-initialize
        st.rerun() # Rerun to reflect changes

# Display Direct Chat History if any
if st.session_state.direct_chat_history:
    with st.expander("💬 Direct LLM Chat History", expanded=True):
        for msg in st.session_state.direct_chat_history:
            st.markdown(f"**{msg['role'].capitalize()}:** {msg['content']}")

if st.session_state.thinking_process_log:
    with st.expander("🧠 Thinking/Reasoning Process", expanded=False):
        st.text_area("Log:", value=st.session_state.thinking_process_log, height=400, key=f"log_display_{len(st.session_state.thinking_process_log)}")

if st.session_state.final_answer_output:
    with st.expander("✅ Final Answer", expanded=True):
        st.markdown(st.session_state.final_answer_output)