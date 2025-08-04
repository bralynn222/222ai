import gradio as gr
import httpx
import json
import asyncio
import re
import io
import contextlib
import traceback
import sys
import os
import uuid
import subprocess  # For running code in a separate process
import tempfile    # For creating temporary files
import signal      # For sending signals to processes (Unix)
import platform    # To check the operating system
import textwrap    # For dedenting code

# --- Configuration ---
DEBUG_MODE = True
CODE_START_DELIMITER = "#C=9ZXP2"
CODE_END_DELIMITER = "#C=9ZXPD"
CODE_EXECUTION_TIMEOUT = 35  # Seconds.
# OUTPUT_TRUNCATION_LENGTH is removed as we now send full output to LLM upon request
API_ENDPOINTS = {
    "OpenAI": "https://api.openai.com/v1",
    "OpenRouter": "https://openrouter.ai/api/v1",
    "Local LLM": "http://localhost:11434"
}
API_KEYS_FILE = "api_keys.json"

# --- Global state for stop signal ---
# Use a dictionary to hold the stop flag, making it mutable and accessible across scopes
stop_signal = {"should_stop": False}

# --- Initial examples for the prompt dropdown ---
INITIAL_EXAMPLES = [
    "What are the first 10 prime numbers? Write code to find out.",
    "Fetch the current price of Bitcoin from the CoinGecko API and print it. The API endpoint is https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd",
    "List all the files in the current directory.",
    # New example to test the timeout
    "Write a script that loops forever and prints a number every second. import time",
    "code the entire game of chess inside of a python script with visual GUI, put all code Between the Keys- #C=9ZXP2 at the start of the code #C=9ZXPD at the end , at all steps Insert a debug print printed to stdout. PROVIDE ONLY PURE CODE NO COMMENTS OR TEXT OF ANY KIND"
]

# --- Static System Prompt ---
SYSTEM_PROMPT = {
    "role": "system",
    "content": f"""You are a powerful AI assistant that can execute Python code to answer questions.
When you need to execute Python code, wrap it in special delimiters.
Start of code: {CODE_START_DELIMITER}
End of code: {CODE_END_DELIMITER}
Insert a debug print printed to stdout for each step in the code to give yourself feedback on your own code"""
}

# --- Functions for Saving and Loading API Keys ---
def save_keys(openai_key: str, openrouter_key: str):
    keys_to_save = {"openai_api_key": openai_key, "openrouter_api_key": openrouter_key}
    try:
        with open(API_KEYS_FILE, "w") as f:
            json.dump(keys_to_save, f, indent=2)
        return "✅ API Keys saved successfully!"
    except Exception as e:
        return f"❌ Error saving keys: {e}"

def load_saved_keys():
    try:
        with open(API_KEYS_FILE, "r") as f:
            keys = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        keys = {}
    openai_key = keys.get("openai_api_key", os.getenv("OPENAI_API_KEY", ""))
    openrouter_key = keys.get("openrouter_api_key", os.getenv("OPENROUTER_API_KEY", ""))
    return openai_key, openrouter_key

# --- Functions for UI Management ---
def add_prompt_to_examples(new_prompt: str, existing_examples: list):
    if not new_prompt or not new_prompt.strip():
        gr.Warning("Cannot save an empty prompt!")
        return existing_examples, gr.update()
    if new_prompt in existing_examples:
        gr.Info("Prompt is already in the examples list.")
        return existing_examples, gr.update(value=new_prompt)
    updated_examples = existing_examples + [new_prompt]
    gr.Success("Saved prompt to examples!")
    return updated_examples, gr.update(choices=updated_examples, value=new_prompt)

def save_conversation_to_sharegpt(history: list):
    if not history:
        gr.Warning("Conversation is empty. Nothing to save.")
        return None
    role_map = {"user": "human", "assistant": "gpt"}
    sharegpt_conversations = []
    for message in history:
        turn = None
        if isinstance(message, dict) and "role" in message:
            turn = message
        elif isinstance(message, (list, tuple)) and len(message) == 2:
            sharegpt_conversations.append({"from": "human", "value": message[0]})
            sharegpt_conversations.append({"from": "gpt", "value": message[1]})
            continue
        else:
            continue
        if turn and "role" in turn and "content" in turn:
            role, content = turn.get("role"), turn.get("content")
            if role in role_map and content:
                sharegpt_conversations.append({"from": role_map[role], "value": content})
    if not sharegpt_conversations:
        gr.Warning("No valid turns found in the conversation to save.")
        return None
    convo_id = str(uuid.uuid4())
    sharegpt_data = [{"id": convo_id, "conversations": sharegpt_conversations}]
    filename = f"sharegpt_conversation_{convo_id}.json"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(sharegpt_data, f, indent=2)
        gr.Info(f"Conversation prepared for download as {filename}")
        return filename
    except Exception as e:
        gr.Error(f"Failed to save conversation file: {e}")
        traceback.print_exc()
        return None

# --- Modified Code Execution using Subprocess ---
def sync_code_executor(code_to_run: str):
    """
    Executes the code in a separate Python subprocess to allow for proper timeout handling
    and capturing output up to the timeout point.
    """
    # --- Robustness Enhancements ---
    # 1. Ensure the code is a string and strip leading/trailing whitespace/newlines
    if not isinstance(code_to_run, str):
        code_to_run = str(code_to_run)
    stripped_code = code_to_run.strip()

    # 2. Handle completely empty code after stripping
    if not stripped_code:
        error_msg = "[SYSTEM_NOTE: An empty or whitespace-only code block was provided for execution.]"
        if DEBUG_MODE:
            print(f"[DEBUG][sync_code_executor] Provided code was empty or whitespace.")
        return error_msg

    # 3. Optionally, dedent the code to handle potential indentation issues
    #    if the LLM includes leading spaces consistently.
    #    This can help if the first line isn't at column 0.
    dedented_code = textwrap.dedent(stripped_code)

    # 4. Log the exact code about to be written for maximum debuggability
    if DEBUG_MODE:
        print(f"[DEBUG][sync_code_executor] === PREPARING CODE FOR EXECUTION ===")
        print(f"[DEBUG][sync_code_executor] Original code length: {len(code_to_run)} chars.")
        print(f"[DEBUG][sync_code_executor] Stripped code length: {len(stripped_code)} chars.")
        print(f"[DEBUG][sync_code_executor] Dedented code length: {len(dedented_code)} chars.")
        # Print first few lines for quick inspection
        code_lines = dedented_code.splitlines()
        for i, line in enumerate(code_lines[:10]): # Show first 10 lines
             print(f"[DEBUG][sync_code_executor] Code Line {i+1}: {repr(line)}")
        if len(code_lines) > 10:
             print(f"[DEBUG][sync_code_executor] ... (and {len(code_lines) - 10} more lines)")
        print(f"[DEBUG][sync_code_executor] === END OF CODE LOG ===")

    # Create a temporary file for the code
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as script_file:
        # Write the *dedented* code to the file
        script_file.write(dedented_code)
        script_path = script_file.name

    if DEBUG_MODE:
        print(f"[DEBUG][sync_code_executor] Starting execution for temp file: {script_path}")

    try:
        # Determine the Python executable (preferably the one running this script)
        python_executable = sys.executable if sys.executable else 'python'
        # Prepare the command to run the script
        cmd = [python_executable, script_path]
        # Use process groups to kill child processes if needed (important for GUIs/spawned processes)
        creationflags = 0
        preexec_fn = None
        if platform.system() == "Windows":
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            preexec_fn = os.setsid # Create a new session and process group
        # Run the subprocess and capture output with a timeout
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=CODE_EXECUTION_TIMEOUT,
            creationflags=creationflags,
            preexec_fn=preexec_fn,
            encoding='utf-8' # Ensure correct encoding
        )
        # Combine stdout and stderr
        full_output = result.stdout + result.stderr
        # Handle case where process succeeded but produced no output
        if not full_output:
             full_output = "[No output was printed, but the code ran without errors (exit code 0).]" if result.returncode == 0 else "[No output captured, but the process finished.]"
        if DEBUG_MODE:
           print(f"[DEBUG][sync_code_executor] Process finished normally. Exit code: {result.returncode}, Output length: {len(full_output)} chars.")
        return full_output
    except subprocess.TimeoutExpired as e:
        # Handle timeout: retrieve output produced so far
        output_capture = io.StringIO()
        if e.stdout:
            # stdout/stderr from TimeoutExpired is bytes, decode it
            output_capture.write(e.stdout.decode('utf-8', errors='replace') if isinstance(e.stdout, bytes) else e.stdout)
        if e.stderr:
            output_capture.write(e.stderr.decode('utf-8', errors='replace') if isinstance(e.stderr, bytes) else e.stderr)
        # Attempt to kill the process group
        if 'result' in locals() and hasattr(result, 'pid'):
            try:
                pid = result.pid
                if platform.system() == "Windows":
                    os.kill(pid, signal.CTRL_BREAK_EVENT) # Or signal.SIGTERM
                else:
                    pgid = os.getpgid(pid)
                    os.killpg(pgid, signal.SIGTERM) # Try graceful termination first
                    # Optionally, wait a bit and then send SIGKILL if it's still running
                    # time.sleep(0.1)
                    # os.killpg(pgid, signal.SIGKILL)
            except ProcessLookupError:
                pass # Process already finished
            except Exception as kill_err:
                if DEBUG_MODE:
                   print(f"[DEBUG][sync_code_executor] Error attempting to kill process on timeout: {kill_err}")
        output_capture.write(
            f"\n[SYSTEM_NOTE: Code execution timed out after {CODE_EXECUTION_TIMEOUT} seconds "
            "and was terminated. The process and any spawned children should have been killed This is by design of the tool used to run code you write it is in no way a problem with your code.]"
        )
        captured_output = output_capture.getvalue()
        output_capture.close()
        if DEBUG_MODE:
            print(f"[DEBUG][sync_code_executor] Execution timed out. Captured output length: {len(captured_output)} chars.")
        return captured_output
    except Exception as e:
        # Handle other errors during subprocess execution itself
        exc_str = traceback.format_exc()
        if DEBUG_MODE:
            print(f"[DEBUG][sync_code_executor] Unexpected error in subprocess execution: {exc_str}")
        # Include the code that failed in the error message for debugging
        error_with_context = (
            f"[SYSTEM_NOTE: An error occurred in the execution host while trying to run the provided code.]\n"
            f"[FAILED_CODE_SNIPPET_START]\n{dedented_code}\n[FAILED_CODE_SNIPPET_END]\n"
            f"[ERROR_DETAILS]\n{str(e)}\n{exc_str}\n[ERROR_DETAILS_END]"
        )
        return error_with_context
    finally:
        # Clean up the temporary script file
        try:
            os.unlink(script_path)
            if DEBUG_MODE:
               print(f"[DEBUG][sync_code_executor] Temporary script file {script_path} deleted.")
        except OSError as e:
            if DEBUG_MODE:
               print(f"[DEBUG][sync_code_executor] Warning: Could not delete temporary file {script_path}: {e}")

# --- Modified Code Extraction and Running Logic (NO PRE-TRUNCATION FOR LLM FEEDBACK) ---
async def extract_and_run_code_with_timeout(llm_output: str):
    """
    Extracts code, runs it using the new subprocess-based executor, enforces a timeout,
    and returns the full output (or output up to timeout).
    This is an async function and must be awaited.
    """
    if DEBUG_MODE:
        print(f"\n[DEBUG][extract_and_run_code_with_timeout] === START ===")
        print(f"[DEBUG][extract_and_run_code_with_timeout] Received LLM output length: {len(llm_output)} chars.")
    pattern = re.compile(f"{re.escape(CODE_START_DELIMITER)}(.*?){re.escape(CODE_END_DELIMITER)}", re.DOTALL)
    match = pattern.search(llm_output)
    if not match:
        if DEBUG_MODE:
            print(f"[DEBUG][extract_and_run_code_with_timeout] No code delimiters found in LLM output.")
        return None, None  # No code found
    extracted_code = match.group(1).strip()
    if not extracted_code:
        if DEBUG_MODE:
            print(f"[DEBUG][extract_and_run_code_with_timeout] Found delimiters, but extracted code is empty.")
        return llm_output, "[SYSTEM_NOTE: An empty code block was provided.]"
    if DEBUG_MODE:
        print(f"[DEBUG][extract_and_run_code_with_timeout] Extracted code length: {len(extracted_code)} chars.")
    loop = asyncio.get_running_loop()
    execution_result = ""
    try:
        # The NEW sync_code_executor handles timeout internally and captures output.
        future = loop.run_in_executor(None, sync_code_executor, extracted_code)
        # Increase the asyncio timeout significantly or remove it, as the subprocess handles the main timeout.
        # Keeping a slightly longer timeout here as a safety net for extreme cases (e.g., process spawn issues).
        execution_result = await asyncio.wait_for(future, timeout=CODE_EXECUTION_TIMEOUT + 10)
        if not execution_result:
            execution_result = "[No output was printed, but the code ran without errors.]"
    except asyncio.TimeoutError:
        # This should rarely be triggered if sync_code_executor works correctly.
        # It means the *subprocess itself* took longer than CODE_EXECUTION_TIMEOUT + 10 to finish/cleanup.
        execution_result = (
            f"[SYSTEM_NOTE: Code execution timed out after {CODE_EXECUTION_TIMEOUT + 10} seconds "
            "at the host level. (Subprocess timeout handled at lower level.)"
        )
    except Exception:
        # This catches errors in the execution management itself (e.g., issues starting the subprocess).
        execution_result = f"An error occurred in the execution host:\n{traceback.format_exc()}"
    # --- IMPORTANT CHANGE: Return the FULL execution result ---
    # The truncation below is ONLY for potential internal logging/display if needed,
    # NOT for the feedback sent to the LLM.
    final_result_for_llm = execution_result.strip()
    # Optional: Truncate only for internal debug prints if it's astronomically large
    # This prevents massive print statements but doesn't affect what the LLM sees.
    debug_result_to_print = final_result_for_llm
    if len(debug_result_to_print) > 100000: # e.g., 100KB limit for debug prints
        half_len = 50000
        truncated_message = (
            f"[DEBUG_TRUNCATED: Output was {len(debug_result_to_print)} chars. "
            f"Showing first/last {half_len} chars.]"
        )
        debug_result_to_print = (
            f"{debug_result_to_print[:half_len].strip()}\n"
            f"{truncated_message}\n"
            f"{debug_result_to_print[-half_len:].strip()}"
        )
    if DEBUG_MODE:
        print(
            f"[DEBUG][extract_and_run_code_with_timeout] Final feedback content length prepared for LLM: {len(final_result_for_llm)} chars.")
        print(f"[DEBUG][extract_and_run_code_with_timeout] Final feedback content (for debug, potentially truncated): ...{debug_result_to_print[-500:] if debug_result_to_print else ''}") # Show last 500 chars for debug
        print(f"[DEBUG][extract_and_run_code_with_timeout] === END ===\n")
    # Return the ORIGINAL, full output for the LLM feedback
    return llm_output, final_result_for_llm

# --- Core Logic: Fetching Models from APIs ---
async def fetch_models(api_provider: str, api_key: str, local_api_url: str):
    if api_provider == "Local LLM":
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{local_api_url}/api/tags")
                response.raise_for_status()
                models = response.json().get("models", [])
                model_names = sorted([m["name"] for m in models])
                return gr.update(choices=model_names,
                                 value=model_names[0] if model_names else None), "✅ Local models loaded."
        except Exception:
            return gr.update(choices=["llama3"], value="llama3"), f"⚠️ Couldn't reach local server."
    if not api_key:
        return gr.update(choices=[], value=None), "⛔️ API Key is required."
    headers = {"Authorization": f"Bearer {api_key}"}
    url = f"{API_ENDPOINTS[api_provider]}/models"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()
            model_ids = sorted([item["id"] for item in response.json().get("data", [])])
            return gr.update(choices=model_ids, value=model_ids[
                0] if model_ids else None), f"✅ {len(model_ids)} {api_provider} models loaded."
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            return gr.update(choices=[], value=None), f"❌ Unauthorized! Check API Key."
        return gr.update(choices=[], value=None), f"❌ HTTP Error {e.response.status_code}."
    except Exception as e:
        return gr.update(choices=[], value=None), f"❌ Error: {e}"

# --- Core Logic: API Interaction and Multi-Step Loop (MODIFIED with STOP CHECKS) ---
async def process_and_stream_response(
        user_prompt: str,
        history: list,
        api_provider: str, model_name: str,
        openai_key: str, openrouter_key: str, local_api_url: str,
        temperature: float, top_p: float, max_tokens: int
):
    # Reset the stop flag at the beginning of a new conversation
    stop_signal["should_stop"] = False

    if not user_prompt:
        yield history, "🟡 Ready"
        return
    if not model_name:
        history.append(
            {"role": "assistant", "content": "**Error:** Model not selected. Please choose a model from the dropdown."})
        yield history, "⛔️ Model not selected!"
        return
    history.append({"role": "user", "content": user_prompt})
    headers, api_key, api_url = {}, None, ""
    if api_provider == "OpenAI":
        api_key, api_url = openai_key, API_ENDPOINTS["OpenAI"] + "/chat/completions"
    elif api_provider == "OpenRouter":
        api_key, api_url = openrouter_key, API_ENDPOINTS["OpenRouter"] + "/chat/completions"
        headers.update({"HTTP-Referer": "http://localhost", "X-Title": "Gradio LLM Agent"})
    else:
        api_url = f"{local_api_url}/v1/chat/completions"
    if api_provider != "Local LLM":
        if not api_key:
            history.append({"role": "assistant", "content": f"**Error:** API Key for {api_provider} is missing."})
            yield history, f"⛔️ Missing {api_provider} API Key!"
            return
        headers["Authorization"] = f"Bearer {api_key}"
    max_loops = 1000
    for loop_count in range(max_loops):
        # --- Check for stop signal before each loop iteration ---
        if stop_signal["should_stop"]:
             yield history, "⏹️ **Stopped by user.**"
             return # Exit the function entirely

        history.append({"role": "assistant", "content": ""})
        yield history, f"🔄 **Loop {loop_count + 1}/{max_loops}:** Sending to {api_provider}..."
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                messages_to_send = [SYSTEM_PROMPT] + history
                payload = {"model": model_name, "messages": messages_to_send, "stream": True,
                           "temperature": temperature, "top_p": top_p}
                if api_provider in ["OpenAI", "OpenRouter"]:
                    payload["max_completion_tokens"] = max_tokens
                else:
                    payload["max_tokens"] = max_tokens
                async with client.stream("POST", api_url, json=payload, headers=headers) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        try:
                            error_details = json.dumps(json.loads(error_body.decode()), indent=2)
                        except json.JSONDecodeError:
                            error_details = error_body.decode()
                        history[-1][
                            "content"] = f"**API Error:** {response.status_code}\n**Details:**\n```json\n{error_details}\n```"
                        yield history, f"❌ API Error: {response.status_code}"
                        return
                    status_text = f"🔄 **Loop {loop_count + 1}:** Streaming from {model_name}..."
                    yield history, status_text
                    async for line in response.aiter_lines():
                        # --- Check for stop signal during streaming ---
                        if stop_signal["should_stop"]:
                             yield history, "⏹️ **Stopped by user.**"
                             return # Exit the function entirely

                        if line.startswith('data: '): # Keep original streaming format from the provided file
                            content = line[len('data: '):].strip()
                            if content == '[DONE]':
                                break
                            try:
                                chunk = json.loads(content)
                                if token := chunk.get("choices", [{}])[0].get("delta", {}).get("content"):
                                    history[-1]["content"] += token
                                    yield history, status_text
                            except (json.JSONDecodeError, IndexError):
                                continue
        except (httpx.RequestError, asyncio.CancelledError) as e:
            # Check if it was cancelled due to the stop mechanism or Gradio event cancellation
            if isinstance(e, asyncio.CancelledError) or stop_signal["should_stop"]:
                 yield history, "⏹️ **Stopped by user.**"
            else:
                 history[-1]["content"] = f"**Network/Client Error:**\n{e}"
                 yield history, "❌ Request Error or Cancelled"
            return
        except Exception as e:
            # Handle other unexpected errors during the request/streaming
            history[-1]["content"] = f"**Unexpected Error:**\n{traceback.format_exc()}"
            yield history, "❌ Unexpected Error"
            return

        # --- Check for stop signal after streaming is done ---
        if stop_signal["should_stop"]:
             yield history, "⏹️ **Stopped by user.**"
             return # Exit the function entirely

        yield history, "🤔 Processing..."
        # --- This is where the feedback content is obtained ---
        llm_response_content, feedback_content = await extract_and_run_code_with_timeout(history[-1]["content"])

        # --- Check for stop signal after code execution ---
        if stop_signal["should_stop"]:
             yield history, "⏹️ **Stopped by user.**"
             return # Exit the function entirely

        # --- Add debug print here to see what's being sent back ---
        if DEBUG_MODE and feedback_content is not None:
            print(
                f"[DEBUG][process_and_stream_response] Feedback content received from extract_and_run_code_with_timeout. Length: {len(feedback_content)} chars.")
            print(
                f"[DEBUG][process_and_stream_response] LLM response content length (for reference): {len(llm_response_content) if llm_response_content else 0} chars.")
        if feedback_content:
            yield history, "✍️ Code executed. Preparing feedback for next loop..."
            await asyncio.sleep(1)
            # --- Modified Feedback Prompt (as requested in the initial query) ---
            feedback_prompt = f"[EXECUTION_FEEDBACK_START]\n{feedback_content}\n[EXECUTION_FEEDBACK_END]\nPlease analyze the output above from your previous code execution. If the task is complete, provide the final answer. If not, write new code between {CODE_START_DELIMITER} and {CODE_END_DELIMITER} to continue. Remember to add debug prints."
            # --- Add debug print here to confirm the prompt being added to history ---
            if DEBUG_MODE:
                print(
                    f"[DEBUG][process_and_stream_response] Adding feedback prompt to history. Length: {len(feedback_prompt)} chars.")
            history.append({"role": "user", "content": feedback_prompt})
            yield history, "🗣️ Starting next loop..."
            await asyncio.sleep(1)
        else:
            yield history, "✅ **Finished:** LLM provided a final answer."
            return
    yield history, f"⚠️ **Max loops reached ({max_loops}).** Stopping."

# --- Function to set the stop signal ---
def set_stop_flag():
    """Function called by the stop button to set the global stop flag."""
    stop_signal["should_stop"] = True
    if DEBUG_MODE:
        print("[DEBUG] Stop flag set to True.")
    # Return a status update message - this will be shown after the event cancellation
    # The main loop checks the flag and yields its own stop message.
    # This return is just for the immediate button click feedback.
    return "⏹️ **Stopping...** (Waiting for current operation to finish)"

# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft(), title="Multi-API LLM Code Agent") as demo:
    examples_state = gr.State(value=INITIAL_EXAMPLES)
    gr.Markdown("# Multi-API LLM Code Agent with Dynamic Models")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### API Configuration")
            api_provider_dropdown = gr.Dropdown(label="API Provider", choices=["Local LLM", "OpenAI", "OpenRouter"],
                                                value="Local LLM")
            local_api_url_textbox = gr.Textbox(label="Local API Endpoint URL", value=API_ENDPOINTS["Local LLM"],
                                               visible=True)
            openai_api_key_textbox = gr.Textbox(label="OpenAI API Key", type="password", visible=False)
            openrouter_api_key_textbox = gr.Textbox(label="OpenRouter API Key", type="password", visible=False)
            save_keys_button = gr.Button("💾 Save API Keys")
            model_name_dropdown = gr.Dropdown(label="Model", choices=[], value=None, interactive=True)
            refresh_models_button = gr.Button("🔄 Get Models")
            with gr.Accordion("⚙️ Generation Parameters", open=False):
                temperature_slider = gr.Slider(0.0, 2.0, 0.7, step=0.1, label="Temperature")
                top_p_slider = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="Top-P")
                max_tokens_slider = gr.Slider(256, 81902, 2048, step=128, label="Max New Tokens")
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=600,
                                 avatar_images=(None, "https://i.imgur.com/b2rQ91v.png"), type="messages")
            status_display = gr.Markdown("🟡 **Ready**", elem_id="status_display")
            with gr.Row():
                prompt_textbox = gr.Textbox(
                    placeholder="e.g., 'What are the first 5 prime numbers? Write code to find out.'", scale=6,
                    container=False)
                send_button = gr.Button("▶️ Send", variant="primary", scale=1, min_width=80)
            with gr.Row():
                examples_dropdown = gr.Dropdown(label="Load an Example", choices=INITIAL_EXAMPLES, interactive=True,
                                                scale=5, container=False, allow_custom_value=False)
                save_example_button = gr.Button("💾 Save Prompt", scale=1, min_width=80)
            with gr.Row():
                stop_button = gr.Button("⏹️ Stop", variant="stop")
                clear_button = gr.Button("🗑️ Clear Chat", variant="secondary")
                save_conversation_button = gr.Button("💾 Save Conversation", variant="secondary")
    download_file_output = gr.File(label="Download Conversation", visible=False, interactive=False)

    def handle_provider_change(provider: str):
        return {local_api_url_textbox: gr.update(visible=provider == "Local LLM"),
                openai_api_key_textbox: gr.update(visible=provider == "OpenAI"),
                openrouter_api_key_textbox: gr.update(visible=provider == "OpenRouter")}

    def get_dynamic_api_key(provider: str, oaik: str, orik: str):
        return oaik if provider == "OpenAI" else orik if provider == "OpenRouter" else ""

    async def handle_refresh_models(provider, oaik, orik, local_url):
        api_key = get_dynamic_api_key(provider, oaik, orik)
        return await fetch_models(provider, api_key, local_url)

    save_keys_button.click(fn=save_keys, inputs=[openai_api_key_textbox, openrouter_api_key_textbox],
                           outputs=[status_display])
    api_provider_dropdown.change(fn=handle_provider_change, inputs=api_provider_dropdown,
                                 outputs=[local_api_url_textbox, openai_api_key_textbox,
                                          openrouter_api_key_textbox]).then(fn=handle_refresh_models,
                                                                            inputs=[api_provider_dropdown,
                                                                                    openai_api_key_textbox,
                                                                                    openrouter_api_key_textbox,
                                                                                    local_api_url_textbox],
                                                                            outputs=[model_name_dropdown,
                                                                                     status_display])
    refresh_models_button.click(fn=handle_refresh_models,
                                inputs=[api_provider_dropdown, openai_api_key_textbox, openrouter_api_key_textbox,
                                        local_api_url_textbox], outputs=[model_name_dropdown, status_display])
    save_example_button.click(fn=add_prompt_to_examples, inputs=[prompt_textbox, examples_state],
                              outputs=[examples_state, examples_dropdown], show_progress="hidden")
    examples_dropdown.change(fn=lambda choice: choice, inputs=examples_dropdown, outputs=prompt_textbox,
                             show_progress="hidden")
    submission_inputs = [prompt_textbox, chatbot, api_provider_dropdown, model_name_dropdown, openai_api_key_textbox,
                         openrouter_api_key_textbox, local_api_url_textbox, temperature_slider, top_p_slider,
                         max_tokens_slider]
    submission_outputs = [chatbot, status_display]
    submit_event_args = {"fn": process_and_stream_response, "inputs": submission_inputs, "outputs": submission_outputs,
                         "show_progress": "hidden", "concurrency_limit": 1}

    # --- Submission Events ---
    # Create a single event that handles both submit and click, allowing cancellation on new submission
    submission_event = gr.on(
        triggers=[prompt_textbox.submit, send_button.click],
        **submit_event_args
    ).then(lambda: "", outputs=prompt_textbox) # Clear textbox after processing starts

    # --- Stop Button ---
    # 1. Set the global stop flag
    # 2. Cancel the ongoing Gradio event (prevents queuing/next iteration)
    stop_button.click(
        fn=set_stop_flag, # Function to set the flag
        inputs=None,
        outputs=[status_display], # Show immediate feedback
        queue=False # Run immediately, don't wait in queue
    ).then( # Then cancel the ongoing event
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submission_event], # Cancel the ongoing event
        queue=False
    )

    # --- Clear Button ---
    clear_button.click(
        fn=lambda: ([], "🟡 **Ready**"),
        inputs=None,
        outputs=[chatbot, status_display],
        queue=False
    ).then( # Also cancel any ongoing event when clearing
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submission_event],
        queue=False
    )
    save_conversation_button.click(fn=save_conversation_to_sharegpt, inputs=[chatbot], outputs=[download_file_output],
                                   show_progress="hidden", queue=False)
    demo.load(fn=load_saved_keys, inputs=None, outputs=[openai_api_key_textbox, openrouter_api_key_textbox]).then(
        fn=handle_refresh_models,
        inputs=[api_provider_dropdown, openai_api_key_textbox, openrouter_api_key_textbox, local_api_url_textbox],
        outputs=[model_name_dropdown, status_display])

if __name__ == "__main__":
    demo.queue().launch(debug=True, show_error=True)