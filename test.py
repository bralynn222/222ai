import gradio as gr
import httpx
import json
import asyncio
import openai
import google.generativeai as genai
from dotenv import load_dotenv
import os
import re
import time
from typing import Optional, Tuple, List, Dict
from datetime import datetime
from functools import partial

try:
    import pandas as pd
except ImportError:
    gr.Warning("Pandas library not found. Judge logging will not be available. Please install with: pip install pandas")
    pd = None

# --- Configuration ---
load_dotenv()
OOBABOOGA_API_URL = os.getenv("OOBABOOGA_API_URL", "http://127.0.0.1:5000/v1")
DEFAULT_OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1"
DEFAULT_LMSTUDIO_API_URL = "http://127.0.0.1:1234/v1"
GEMINI_SAFETY_SETTINGS = [  # Lenient safety settings
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# --- Settings File ---
SETTINGS_FILE = "playground_settings.json"

# --- Judge Logic File Paths ---
INSTANCE_ID = "advanced_chat_playground"
DPO_FILE = f"dpo_dataset_{INSTANCE_ID}.jsonl"
LOG_FILE = f"judgement_log_{INSTANCE_ID}.jsonl"

if not os.path.exists(DPO_FILE):
    with open(DPO_FILE, 'w', encoding='utf-8') as f: pass
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', encoding='utf-8') as f: pass

# --- Global Loop Control Flag for Judge Loop ---
JUDGE_AUTO_LOOP_ACTIVE = False


# --- Settings Helper Functions ---
def load_settings() -> Dict:
    """Loads settings from the JSON file."""
    default_settings = {
        "api_keys": {
            "or_key_a": "", "or_key_b": "", "gemini_key_a": "", "gemini_key_b": "",
            "judge_openai_api_key": "", "judge_or_api_key": "", "judge_gemini_api_key": ""
        },
        "prompts": {
            "manual": [],
            "loop": []
        }
    }
    if not os.path.exists(SETTINGS_FILE):
        return default_settings
    try:
        with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
            settings = json.load(f)
            # Ensure all keys from default are present to avoid errors on updates
            for key, value in default_settings.items():
                if key not in settings:
                    settings[key] = value
                elif isinstance(value, dict):
                    for sub_key, sub_value in value.items():
                        if sub_key not in settings[key]:
                            settings[key][sub_key] = sub_value
            return settings
    except (json.JSONDecodeError, IOError):
        return default_settings


def save_settings(settings: Dict):
    """Saves settings to the JSON file."""
    try:
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=4)
        gr.Info("Settings saved!")
    except IOError as e:
        gr.Warning(f"Failed to save settings: {e}")


def save_api_keys(or_a, or_b, gem_a, gem_b, judge_oa, judge_or, judge_gem):
    """Loads settings, updates all API keys, and saves back."""
    settings = load_settings()
    settings["api_keys"]["or_key_a"] = or_a if or_a else ""
    settings["api_keys"]["or_key_b"] = or_b if or_b else ""
    settings["api_keys"]["gemini_key_a"] = gem_a if gem_a else ""
    settings["api_keys"]["gemini_key_b"] = gem_b if gem_b else ""
    settings["api_keys"]["judge_openai_api_key"] = judge_oa if judge_oa else ""
    settings["api_keys"]["judge_or_api_key"] = judge_or if judge_or else ""
    settings["api_keys"]["judge_gemini_api_key"] = judge_gem if judge_gem else ""
    save_settings(settings)


def save_new_prompt(prompt_type: str, text: str):
    """Saves a new prompt to the list if it's unique and not empty."""
    if not text or not text.strip():
        gr.Warning("Cannot save an empty prompt.")
        settings = load_settings()
        prompt_list = settings.get("prompts", {}).get(prompt_type, [])
        return gr.update(choices=prompt_list)

    settings = load_settings()
    if "prompts" not in settings: settings["prompts"] = {"manual": [], "loop": []}
    if prompt_type not in settings["prompts"]: settings["prompts"][prompt_type] = []

    prompt_list = settings["prompts"][prompt_type]
    if text not in prompt_list:
        prompt_list.insert(0, text)
        settings["prompts"][prompt_type] = prompt_list[:50]  # Limit to 50 saved prompts
        save_settings(settings)
    else:
        gr.Info("Prompt is already in the list.")

    return gr.update(choices=settings["prompts"][prompt_type], value=text)


# --- API Helper Functions ---
def unload_local_model(model_name: str):
    if not model_name: return "No model to unload."
    try:
        response = httpx.post(f"{OOBABOOGA_API_URL}/internal/model/unload", json={"model_name": model_name},
                              timeout=120)
        response.raise_for_status()
        return f"Unloaded '{model_name}'."
    except Exception as e:
        return f"Error during unload: {e}"


def load_local_model(model_name: str, load_in_4bit: bool):
    if not model_name: return "No model selected to load.", None
    try:
        payload = {"model_name": model_name, "args": {"load-in-4bit": load_in_4bit}}
        response = httpx.post(f"{OOBABOOGA_API_URL}/internal/model/load", json=payload, timeout=300)
        response.raise_for_status()
        if response.text and "already loaded" in response.text.lower():
            return f"Model '{model_name}' was already loaded.", model_name
        return f"✅ Successfully loaded '{model_name}'.", model_name
    except Exception as e:
        return f"❌ Error loading model '{model_name}': {e}", None


async def get_openrouter_models(api_key: str):
    if not api_key: return []
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{DEFAULT_OPENROUTER_API_URL}/models", headers=headers, timeout=30)
            response.raise_for_status()
            models_data = response.json().get('data', [])
            return sorted([model['id'] for model in models_data if isinstance(model.get('id'), str)])
    except Exception as e:
        gr.Warning(f"Could not fetch OpenRouter models. Check API key/connection. Error: {e}")
        return []


def get_local_models():
    try:
        response = httpx.get(f"{OOBABOOGA_API_URL}/internal/model/list", timeout=20);
        response.raise_for_status()
        data = response.json();
        model_names = data.get("model_names", []) if isinstance(data, dict) else data
        return sorted([m for m in model_names if isinstance(m, str)]) if isinstance(model_names, list) else []
    except Exception as e:
        gr.Warning(f"Could not connect to Oobabooga API. Error: {e}");
        return []


def get_gemini_models():
    """Returns a static list of popular Gemini models."""
    return ["gemini-1.5-flash-latest", "gemini-1.5-pro-latest", "gemini-1.0-pro", "gemini-pro"]


def convert_history_to_gemini(history: List[Dict]) -> List[Dict]:
    """Converts OpenAI-style history to Gemini-style, merging consecutive user messages."""
    gemini_history = []
    if not history:
        return []

    # Iterate through a copy of the history to avoid modification issues
    temp_history = list(history)
    for i in range(len(temp_history)):
        # Skip assistant messages
        if temp_history[i]['role'] == 'assistant':
            # Gemini history must alternate roles. Add the assistant response.
            # The model's role in Gemini is 'model', not 'assistant'.
            gemini_history.append({'role': 'model', 'parts': [temp_history[i]['content']]})
            continue

        # If it's a user message, process it
        if temp_history[i]['role'] == 'user':
            # Merge with the previous message if it was also a 'user' message
            if gemini_history and gemini_history[-1]['role'] == 'user':
                gemini_history[-1]['parts'][-1] += "\n\n" + temp_history[i]['content']
            else:
                gemini_history.append({'role': 'user', 'parts': [temp_history[i]['content']]})
    return gemini_history


async def stream_gemini_chat(api_key: str, model: str, messages: list, params: dict):
    """Streams response from Google Gemini."""
    if not api_key:
        yield "\n\n**ERROR**: Google AI Studio API Key is missing."
        return

    try:
        genai.configure(api_key=api_key)
        # Use the GenerativeModel class
        gemini_model = genai.GenerativeModel(
            model_name=model,
            generation_config=params,
            safety_settings=GEMINI_SAFETY_SETTINGS
        )

        # Convert history and get the latest user message
        gemini_history = convert_history_to_gemini(messages[:-1])
        user_message = messages[-1]['content']

        # Start the chat session with the converted history
        chat_session = gemini_model.start_chat(history=gemini_history)

        # Send the new message and stream the response
        response = chat_session.send_message(user_message, stream=True)

        for chunk in response:
            if chunk.text:
                yield chunk.text
            await asyncio.sleep(0.001)

    except asyncio.CancelledError:
        raise
    except Exception as e:
        yield f"\n\n**ERROR**: Gemini API Error ({model}): {e}"


async def stream_local_chat(model: str, messages: list, params: dict):
    payload = {"model": model, "messages": messages, "stream": True, **params}
    try:
        async with httpx.AsyncClient() as client:
            async with client.stream("POST", f"{OOBABOOGA_API_URL}/chat/completions", json=payload,
                                     timeout=300) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith('data: '):
                        content = line[len('data: '):];
                        if content.strip() == '[DONE]': break
                        try:
                            chunk = json.loads(content);
                            token = chunk.get("choices", [{}])[0].get("delta", {}).get("content")
                            if token: yield token
                        except json.JSONDecodeError:
                            continue
                    await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        raise
    except Exception as e:
        yield f"\n\n**ERROR**: Local model ({model}). {e}"


async def stream_openrouter_chat(api_key: str, model: str, messages: list, params: dict):
    if not api_key: yield "\n\n**ERROR**: OpenRouter API Key missing for generation."; return
    client = openai.AsyncOpenAI(api_key=api_key, base_url=DEFAULT_OPENROUTER_API_URL)
    try:
        stream = await client.chat.completions.create(model=model, messages=messages, stream=True, **params)
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token: yield token
            await asyncio.sleep(0.001)
    except asyncio.CancelledError:
        raise
    except openai.APIError as e:
        yield f"\n\n**ERROR**: OpenRouter API Error ({model}): {type(e).__name__} - {e.message if hasattr(e, 'message') else e}"
    except Exception as e:
        yield f"\n\n**ERROR**: OpenRouter ({model}). {e}"


def get_gemini_api_response_for_judge(api_key: str, model: str, prompt: str, max_tokens: int = 4096):
    """Gets a non-streaming response from Gemini for judging purposes."""
    if not api_key:
        return "[JUDGE API ERROR: Gemini API Key is missing.]"
    try:
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(
            model_name=model,
            generation_config={"max_output_tokens": max_tokens},
            safety_settings=GEMINI_SAFETY_SETTINGS
        )
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"[JUDGE API ERROR: {e}]"


def get_openai_api_response(api_url: str, api_key: str, model: str, prompt: str, max_tokens: int = 2048,
                            extra_headers: Optional[dict] = None):
    final_api_url = api_url.rstrip('/');
    if not ("/v1" in final_api_url or "/api/" in final_api_url): final_api_url += "/v1"
    try:
        client = openai.OpenAI(api_key=api_key or "NA", base_url=final_api_url)
        params_judge = {"model": model, "messages": [{"role": "user", "content": prompt}]}
        is_local_ooba = "127.0.0.1" in final_api_url or "localhost" in final_api_url
        if not is_local_ooba:
            params_judge["max_tokens"] = max_tokens
        response = client.chat.completions.create(**params_judge, extra_headers=extra_headers or {})
        return response.choices[0].message.content
    except Exception as e:
        error_str = str(e).lower()
        if not is_local_ooba and "max_tokens" in params_judge and "max_tokens" in error_str and (
                "unsupported_parameter" in error_str or "extra_body" in error_str or "additional properties" in error_str or "unexpected keyword argument 'max_new_tokens'" in error_str):
            try:
                params_judge.pop("max_tokens", None)
                response = client.chat.completions.create(**params_judge, extra_headers=extra_headers or {})
                return response.choices[0].message.content
            except Exception as retry_e:
                return f"[JUDGE API RETRY ERROR: {retry_e}]"
        return f"[JUDGE API ERROR: {e}]"


def save_preference(prompt: str, response_a: str, response_b: str, choice: str, comment: str, dpo_file: str,
                    log_file: str):
    response_a, response_b, comment, prompt = (str(x) if x is not None else "" for x in
                                               [response_a, response_b, comment, prompt])
    if not prompt and (not response_a or not response_b): return "Cannot save. Prompt missing & responses empty.", False
    log_record = {"timestamp": time.time(), "prompt": prompt, "response_a": response_a, "response_b": response_b,
                  "choice": choice, "comment": comment}
    try:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_record) + "\n")
    except Exception as e:
        return f"Error writing to log file {log_file}: {e}", False
    if choice in ["A is better", "B is better"]:
        chosen, rejected = (response_a, response_b) if choice == "A is better" else (response_b, response_a)
        if not chosen.strip() or not rejected.strip(): return f"Logged '{choice}'. Not saved to DPO (empty response).", True
        dpo_record = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
        try:
            with open(dpo_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(dpo_record) + "\n")
            return f"Saved preference to {dpo_file} and {log_file}!", True
        except Exception as e:
            return f"Error writing to DPO file {dpo_file}: {e}", False
    return f"Logged '{choice}' to UI log ({log_file}). Not saved to DPO dataset.", True


def load_and_format_log(log_file: str):
    if pd is None: return "Pandas library not available."
    log_entries = []
    if not os.path.exists(log_file): return pd.DataFrame([], columns=["Timestamp", "Winner", "Prompt Snippet",
                                                                      "Rationale Snippet"])
    with open(log_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                log_entries.append({
                    "Timestamp": datetime.fromtimestamp(data.get("timestamp", 0)).strftime('%Y-%m-%d %H:%M:%S'),
                    "Winner": data.get("choice", "Unknown"),
                    "Prompt Snippet": data.get("prompt", "")[:100] + (
                        '...' if len(data.get("prompt", "")) > 100 else ''),
                    "Rationale Snippet": data.get("comment", "N/A")[:150] + (
                        '...' if len(data.get("comment", "N/A")) > 150 else ''),
                })
            except Exception:
                continue
    log_entries.reverse()
    return pd.DataFrame(log_entries) if log_entries else pd.DataFrame([],
                                                                      columns=["Timestamp", "Winner", "Prompt Snippet",
                                                                               "Rationale Snippet"])


def auto_judge_logic(prompt: str, response_a: str, response_b: str, judge_type: str, dpo_fp: str, log_fp: str,
                     judge_response_text: str, judge_model_name: str) -> Tuple[str, str, str]:
    """Parses judge response, determines winner, and saves results."""
    if "[JUDGE API ERROR:" in judge_response_text or "[JUDGE API RETRY ERROR:" in judge_response_text:
        return judge_response_text, judge_response_text, "Error"

    rationale = f"AUTO-JUDGE ({judge_model_name}):\n{judge_response_text}"
    match = re.search(r'Key-WINBOY=\s*\(?\s*(A|B|999)\s*\)?', judge_response_text,
                      re.IGNORECASE | re.DOTALL)

    prelim_choice, save_status = "Error", "Key-WINBOY not found in judge response."
    if match:
        result = match.group(1).upper()
        if result == 'A':
            prelim_choice = "A is better"
        elif result == 'B':
            prelim_choice = "B is better"
        elif result == '999':
            prelim_choice = "Both are 999 / Tie"
        save_status, _ = save_preference(prompt, response_a, response_b, prelim_choice, rationale, dpo_fp, log_fp)

    return save_status, rationale, prelim_choice


async def call_auto_judge_interface(last_user_prompt: str, history_a: list, history_b: list, judge_type: str,
                                    current_ooba_model: str,
                                    judge_openai_api_url: str, judge_openai_api_key: str, judge_openai_model: str,
                                    judge_or_api_url: str, judge_or_api_key: str, judge_or_model: str,
                                    judge_or_http_referer: str, judge_or_x_title: str,
                                    judge_lmstudio_api_url: str, judge_lmstudio_model: str,
                                    judge_gemini_api_key: str, judge_gemini_model: str):
    prompt_to_judge = last_user_prompt
    if not prompt_to_judge:
        if history_a and history_a[-1]['role'] == 'user':
            prompt_to_judge = history_a[-1]['content']
        elif history_a and len(history_a) > 1 and history_a[-2]['role'] == 'user':
            prompt_to_judge = history_a[-2]['content']
        else:
            prompt_to_judge = "User prompt not found in history."
    response_a = history_a[-1]['content'] if history_a and history_a[-1][
        'role'] == 'assistant' else "(Response A missing)"
    response_b = history_b[-1]['content'] if history_b and history_b[-1][
        'role'] == 'assistant' else "(Response B missing)"

    if response_a.startswith("(Response A missing") and response_b.startswith("(Response B missing"):
        return "No assistant responses to judge.", "", "Error", load_and_format_log(LOG_FILE)

    judge_prompt_template = (
        f'You are an impartial AI assistant. Your task is to evaluate two responses (A and B) based on a user prompt. '
        f'Your evaluation should be based on adherence to the user\'s request (75% weight) and the accuracy and depth of the information (25% weight). '
        f'You may also consider factors like clarity, conciseness, and formatting.\n\n'
        f'First, provide a brief but clear rationale for your decision, explaining your reasoning for the choice. '
        f'Then, on a new line, you MUST state the winner using ONE of the following formats:\n'
        f'   - "Key-WINBOY=A" if Response A is better.\n'
        f'   - "Key-WINBOY=B" if Response B is better.\n'
        f'   - "Key-WINBOY=999" if BOTH responses are of poor quality, fail to follow the prompt\'s instructions, or are otherwise incorrect. Do NOT choose a winner in this case.\n\n'
        f'Do not deviate from these key formats. Your entire response must contain your reasoning followed by the key.\n\n'
        f'--- USER PROMPT ---\n{prompt_to_judge}\n\n--- RESPONSE A ---\n{response_a}\n\n'
        f'--- RESPONSE B ---\n{response_b}\n\n--- YOUR EVALUATION AND CHOICE ---'
    )

    judge_response, model_for_judge = "", "Unknown"

    if judge_type == "Oobabooga (Local Judge)":
        if not current_ooba_model: return "Error: No Ooba model loaded for judge.", "", "Error", load_and_format_log(
            LOG_FILE)
        model_for_judge = current_ooba_model
        judge_response = get_openai_api_response(OOBABOOGA_API_URL, "NA", model_for_judge, judge_prompt_template)
    elif judge_type == "OpenAI / Cloud API":
        if not judge_openai_model: return "Error: OpenAI Judge Model not selected.", "", "Error", load_and_format_log(
            LOG_FILE)
        model_for_judge = judge_openai_model
        judge_response = get_openai_api_response(judge_openai_api_url, judge_openai_api_key, model_for_judge,
                                                 judge_prompt_template)
    elif judge_type == "OpenRouter":
        if not judge_or_model: return "Error: OpenRouter Judge Model not selected.", "", "Error", load_and_format_log(
            LOG_FILE)
        model_for_judge = judge_or_model
        headers = {h: v for h, v in [("HTTP-Referer", judge_or_http_referer), ("X-Title", judge_or_x_title)] if v}
        judge_response = get_openai_api_response(judge_or_api_url, judge_or_api_key, model_for_judge,
                                                 judge_prompt_template, extra_headers=headers)
    elif judge_type == "LM Studio":
        if not judge_lmstudio_model: return "Error: LM Studio Judge Model not specified.", "", "Error", load_and_format_log(
            LOG_FILE)
        model_for_judge = judge_lmstudio_model
        judge_response = get_openai_api_response(judge_lmstudio_api_url, "NA", model_for_judge, judge_prompt_template)
    elif judge_type == "Google Gemini":
        if not judge_gemini_model: return "Error: Gemini Judge Model not selected.", "", "Error", load_and_format_log(
            LOG_FILE)
        model_for_judge = judge_gemini_model
        judge_response = get_gemini_api_response_for_judge(judge_gemini_api_key, model_for_judge, judge_prompt_template)
    else:
        return f"Error: Unknown judge type '{judge_type}'", "", "Error", load_and_format_log(LOG_FILE)

    status, rationale, choice = auto_judge_logic(prompt_to_judge, response_a, response_b, judge_type, DPO_FILE,
                                                 LOG_FILE, judge_response, model_for_judge)
    return status, rationale, choice, load_and_format_log(LOG_FILE)


def list_api_models_for_judge(api_url: str, api_key: str):
    if not api_url: return gr.update(choices=[], value=None), "[Error] API URL required."
    final_api_url = api_url.rstrip('/');
    if not ("/v1" in final_api_url or "/api/" in final_api_url): final_api_url += "/v1"
    try:
        client = openai.OpenAI(api_key=api_key or "NA", base_url=final_api_url)
        models = sorted([m.id for m in client.models.list() if isinstance(m.id, str)])
        if not models: return gr.update(choices=[], value=None, interactive=True), "Connected, but no models found."
        common_defaults = ["gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "claude-3-opus", "claude-3-sonnet",
                           "deepseek/deepseek-chat", "mistralai/mixtral-8x7b-instruct", "google/gemini-pro"]
        selected_model = models[0]
        for pref in common_defaults:
            if pref in models: selected_model = pref; break
            if any(m.endswith(f"/{pref}") for m in models): selected_model = next(
                m for m in models if m.endswith(f"/{pref}")); break
            if any(pref in m for m in models): selected_model = next(m for m in models if pref in m); break
        return gr.update(choices=models, value=selected_model,
                         interactive=True), f"Success! Found {len(models)} models."
    except Exception as e:
        return gr.update(choices=[], value=None, interactive=True), f"[Error listing models for judge: {e}]"


def create_chat_column(label: str):
    with gr.Column():
        gr.Markdown(f"### {label}")
        source_select = gr.Radio(["OpenRouter", "Google Gemini", "Local (Oobabooga)"], label="Model Source",
                                 value="OpenRouter")
        with gr.Group(visible=True) as or_group:
            or_api_key = gr.Textbox(label="OpenRouter API Key", type="password", lines=1)
        with gr.Group(visible=False) as gemini_group:
            gemini_api_key = gr.Textbox(label="Google AI Studio API Key", type="password", lines=1)
        model_dropdown = gr.Dropdown(label="Select Model", interactive=True, allow_custom_value=True)
        max_tokens_slider_ui = gr.Slider(128, 8192, 1024, step=128, label="Max Tokens")
        chatbot = gr.Chatbot(label=label, height=500, avatar_images=("🧑", "🤖"), type="messages")
    return source_select, or_api_key, gemini_api_key, or_group, gemini_group, model_dropdown, max_tokens_slider_ui, chatbot


async def update_model_list(source: str, or_key: str, gemini_key: str):
    allow_custom, models, vis_or, vis_gemini = False, [], False, False
    selected_value = None

    if source == "OpenRouter":
        models = await get_openrouter_models(or_key)
        vis_or, allow_custom = True, True
        if models:
            selected_value = models[0]
            common_or_defaults = ["deepseek/deepseek-coder", "mistralai/mistral-7b-instruct", "openai/gpt-3.5-turbo",
                                  "google/gemini-1.5-flash-latest", "anthropic/claude-3-haiku",
                                  "deepseek/deepseek-chat"]
            for d in common_or_defaults:
                if d in models: selected_value = d; break
            if selected_value == models[0]:
                for d in common_or_defaults:
                    if any(d in m for m in models): selected_value = next(m for m in models if d in m); break
    elif source == "Google Gemini":
        models = get_gemini_models()
        vis_gemini, allow_custom = True, True
        if models: selected_value = "gemini-1.5-flash-latest"
    elif source == "Local (Oobabooga)":
        models = get_local_models()
        if models: selected_value = models[0]

    return gr.update(choices=models, value=selected_value, interactive=True,
                     allow_custom_value=allow_custom), gr.update(visible=vis_or), gr.update(visible=vis_gemini)


async def run_chat_streams(
        user_message: str,
        history_a: list, source_a: str, or_key_a: str, gemini_key_a: str, model_a: str, slider_max_tokens_a: int,
        history_b: list, source_b: str, or_key_b: str, gemini_key_b: str, model_b: str, slider_max_tokens_b: int,
        temp: float, top_p: float,
        current_local_model: str,
):
    if not user_message or user_message.strip() == "":
        current_prompt_for_judge = user_message if user_message else (
            history_a[-1]['content'] if history_a and history_a[-1]['role'] == 'user' else (
                history_b[-1]['content'] if history_b and history_b[-1]['role'] == 'user' else ""))
        yield history_a, history_b, current_prompt_for_judge, history_a, history_b;
        return

    # Warning for local model not loaded
    warn_local_not_loaded = False
    if (source_a == "Local (Oobabooga)" and model_a and model_a != current_local_model) or \
            (source_b == "Local (Oobabooga)" and model_b and model_b != current_local_model):
        gr.Warning(
            f"A selected local model is not loaded ('{current_local_model or 'None'}'). Load it, then send again.")
        yield history_a, history_b, user_message, history_a, history_b;
        return

    current_history_a, current_history_b = list(history_a), list(history_b)
    generators_data = []

    def prepare_generator(source, model, messages, api_key_or, api_key_gemini, max_tokens, temp, top_p, gen_id):
        if not model: return None

        # Copy message history to avoid mutation
        api_messages = list(messages)
        api_messages.append({"role": "user", "content": user_message})

        # Append user message and empty assistant placeholder to the display history
        display_history = list(messages)
        display_history.append({"role": "user", "content": user_message})
        display_history.append({"role": "assistant", "content": ""})

        params = {"temperature": temp, "top_p": top_p, "max_tokens": max_tokens}

        gen_instance = None
        if source == "OpenRouter":
            gen_instance = stream_openrouter_chat(api_key_or, model, api_messages, params)
        elif source == "Google Gemini":
            # Gemini generation config doesn't use 'max_tokens' directly
            gemini_params = {"temperature": temp, "top_p": top_p, "max_output_tokens": max_tokens}
            gen_instance = stream_gemini_chat(api_key_gemini, model, api_messages, gemini_params)
        elif source == "Local (Oobabooga)":
            gen_instance = stream_local_chat(model, api_messages, params)

        if gen_instance:
            return {'gen_instance': gen_instance, 'hist_ref': display_history, 'id': gen_id}
        return None

    gen_a_data = prepare_generator(source_a, model_a, history_a, or_key_a, gemini_key_a, slider_max_tokens_a, temp,
                                   top_p, 'A')
    gen_b_data = prepare_generator(source_b, model_b, history_b, or_key_b, gemini_key_b, slider_max_tokens_b, temp,
                                   top_p, 'B')

    if gen_a_data:
        current_history_a = gen_a_data['hist_ref']
        generators_data.append(gen_a_data)
    if gen_b_data:
        current_history_b = gen_b_data['hist_ref']
        generators_data.append(gen_b_data)

    if not generators_data:
        yield history_a, history_b, user_message, history_a, history_b
        return

    async def stream_wrapper(gen_instance, target_hist_list, stream_id):
        try:
            async for token in gen_instance:
                if target_hist_list and target_hist_list[-1]["role"] == "assistant":
                    target_hist_list[-1]["content"] += token
                yield
                await asyncio.sleep(0.001)
        except asyncio.CancelledError:
            if target_hist_list and target_hist_list[-1]["role"] == "assistant" and not target_hist_list[-1][
                "content"].endswith("STOPPED"):
                target_hist_list[-1]["content"] += "\n**STREAM STOPPED**"
            raise
        except Exception as e:
            if target_hist_list and target_hist_list[-1]["role"] == "assistant":
                target_hist_list[-1]["content"] += f"\n**ERR {stream_id}**: {e}"
            yield

    active_wrapped_gens = [stream_wrapper(data['gen_instance'], data['hist_ref'], data['id']) for data in
                           generators_data]
    try:
        while active_wrapped_gens:
            for wrapped_gen in active_wrapped_gens[:]:
                try:
                    await wrapped_gen.__anext__()
                except StopAsyncIteration:
                    active_wrapped_gens.remove(wrapped_gen)
                except asyncio.CancelledError:
                    active_wrapped_gens.clear()  # Stop all generators
                    break

            yield current_history_a, current_history_b, user_message, current_history_a, current_history_b
            if not active_wrapped_gens: break
            await asyncio.sleep(0.01)

    except asyncio.CancelledError:
        pass  # The wrapper handles appending the STOPPED message

    yield current_history_a, current_history_b, user_message, current_history_a, current_history_b


async def auto_judge_loop_runner(
        judge_loop_prompt: str,
        source_a_gen: str, or_key_a_gen: str, gemini_key_a_gen: str, model_a_gen: str, slider_max_tokens_gen_a: int,
        source_b_gen: str, or_key_b_gen: str, gemini_key_b_gen: str, model_b_gen: str, slider_max_tokens_gen_b: int,
        temp_gen: float, top_p_gen: float, current_ooba_model_gen: str,
        judge_type_cfg: str, judge_openai_api_url_cfg: str, judge_openai_api_key_cfg: str, judge_openai_model_cfg: str,
        judge_or_api_url_cfg: str, judge_or_api_key_cfg: str, judge_or_model_cfg: str,
        judge_or_http_referer_cfg: str, judge_or_x_title_cfg: str,
        judge_lmstudio_api_url_cfg: str, judge_lmstudio_model_cfg: str,
        judge_gemini_api_key_cfg: str, judge_gemini_model_cfg: str,
        delay_seconds: int
):
    global JUDGE_AUTO_LOOP_ACTIVE;
    JUDGE_AUTO_LOOP_ACTIVE = True;
    cycle_count = 0
    if not judge_loop_prompt or judge_loop_prompt.strip() == "":
        yield ("Loop Error: Loop prompt is empty.", None, None, "", None, None, None, None, "Error",
               load_and_format_log(LOG_FILE));
        JUDGE_AUTO_LOOP_ACTIVE = False;
        return

    latest_chatbot_hist_a, latest_chatbot_hist_b = [], []
    judge_rationale_val, error_msg_val = "", ""

    while JUDGE_AUTO_LOOP_ACTIVE:
        cycle_count += 1;
        gen_status_msg = f"Loop {cycle_count}: Generating \"{judge_loop_prompt[:30]}...\""
        yield (gen_status_msg, [], [], "", None, None, None, None, "Error", load_and_format_log(LOG_FILE))

        current_run_hist_a, current_run_hist_b = [], []

        try:
            async for stream_hist_a, stream_hist_b, _, _, _ in run_chat_streams(
                    user_message=judge_loop_prompt,
                    history_a=[], source_a=source_a_gen, or_key_a=or_key_a_gen, gemini_key_a=gemini_key_a_gen,
                    model_a=model_a_gen,
                    slider_max_tokens_a=slider_max_tokens_gen_a,
                    history_b=[], source_b=source_b_gen, or_key_b=or_key_b_gen, gemini_key_b=gemini_key_b_gen,
                    model_b=model_b_gen,
                    slider_max_tokens_b=slider_max_tokens_gen_b,
                    temp=temp_gen, top_p=top_p_gen, current_local_model=current_ooba_model_gen
            ):
                current_run_hist_a, current_run_hist_b = stream_hist_a, stream_hist_b
                yield (gen_status_msg, current_run_hist_a, current_run_hist_b, judge_rationale_val, None, None, None,
                       None, "Error", load_and_format_log(LOG_FILE))
                if not JUDGE_AUTO_LOOP_ACTIVE: break
                await asyncio.sleep(0.01)

            if not JUDGE_AUTO_LOOP_ACTIVE: break

            latest_chatbot_hist_a, latest_chatbot_hist_b = current_run_hist_a, current_run_hist_b

            # Check for generation errors
            for hist, name in [(latest_chatbot_hist_a, "A"), (latest_chatbot_hist_b, "B")]:
                if hist and hist[-1]['role'] == 'assistant' and "**ERROR**" in hist[-1].get('content', ''):
                    raise Exception(f"Generation {name} Error: {hist[-1]['content']}")

        except Exception as e:
            error_msg_val = f"Loop Cycle {cycle_count}: Generation failed: {e}. Stopping."
            yield (error_msg_val, current_run_hist_a, current_run_hist_b, judge_rationale_val, None, None, None, None,
                   "Error", load_and_format_log(LOG_FILE))
            JUDGE_AUTO_LOOP_ACTIVE = False;
            break
        if not JUDGE_AUTO_LOOP_ACTIVE: break

        judge_status_update = f"Loop {cycle_count}: Judging..."
        yield (judge_status_update, latest_chatbot_hist_a, latest_chatbot_hist_b, judge_rationale_val, None, None, None,
               None, "Error", load_and_format_log(LOG_FILE))
        if not JUDGE_AUTO_LOOP_ACTIVE: break

        try:
            judge_status, judge_rationale_val, judge_prelim_choice, judge_log_df = await call_auto_judge_interface(
                judge_loop_prompt, latest_chatbot_hist_a, latest_chatbot_hist_b,
                judge_type_cfg, current_ooba_model_gen,
                judge_openai_api_url_cfg, judge_openai_api_key_cfg, judge_openai_model_cfg,
                judge_or_api_url_cfg, judge_or_api_key_cfg, judge_or_model_cfg,
                judge_or_http_referer_cfg, judge_or_x_title_cfg,
                judge_lmstudio_api_url_cfg, judge_lmstudio_model_cfg,
                judge_gemini_api_key_cfg, judge_gemini_model_cfg
            )
            if not JUDGE_AUTO_LOOP_ACTIVE: break
            if "Error" in judge_status or "[JUDGE API ERROR:" in judge_status or "[JUDGE API RETRY ERROR:" in judge_status:
                error_msg_val = f"Loop {cycle_count} Judge err: {judge_status}. Stop."
                yield (error_msg_val, latest_chatbot_hist_a, latest_chatbot_hist_b, judge_rationale_val, None, None,
                       None, None, "Error", judge_log_df)
                JUDGE_AUTO_LOOP_ACTIVE = False;
                break
            if not JUDGE_AUTO_LOOP_ACTIVE: break
            ui_status = f"Loop {cycle_count}: {judge_status}"
            yield (ui_status, latest_chatbot_hist_a, latest_chatbot_hist_b, judge_rationale_val, None, None, None, None,
                   judge_prelim_choice, judge_log_df)
        except Exception as e:
            error_msg_val = f"Loop {cycle_count} Judge process fail: {e}. Stop."
            yield (error_msg_val, latest_chatbot_hist_a, latest_chatbot_hist_b, "", None, None, None, None, "Error",
                   load_and_format_log(LOG_FILE))
            JUDGE_AUTO_LOOP_ACTIVE = False;
            break
        if not JUDGE_AUTO_LOOP_ACTIVE: break

        delay_msg = f"Loop {cycle_count} done. Pause {delay_seconds}s..."
        yield (delay_msg, latest_chatbot_hist_a, latest_chatbot_hist_b, judge_rationale_val, None, None, None, None,
               judge_prelim_choice, load_and_format_log(LOG_FILE))
        await asyncio.sleep(delay_seconds)

    final_status = "Loop finished."
    if not JUDGE_AUTO_LOOP_ACTIVE and cycle_count > 0:
        final_status = error_msg_val if error_msg_val else "Loop stopped by user or completed."
    elif cycle_count == 0 and not JUDGE_AUTO_LOOP_ACTIVE:
        final_status = "Loop stopped pre-start."
    yield (final_status, latest_chatbot_hist_a, latest_chatbot_hist_b, judge_rationale_val, None, None, None, None,
           "Error", load_and_format_log(LOG_FILE))
    JUDGE_AUTO_LOOP_ACTIVE = False


def stop_judge_loop_global():
    global JUDGE_AUTO_LOOP_ACTIVE
    if JUDGE_AUTO_LOOP_ACTIVE: JUDGE_AUTO_LOOP_ACTIVE = False; return "Loop stop signal sent."
    return "Loop not active."


def handle_local_model_selection_change(source_dropdown_val: str, selected_model_in_dropdown: str,
                                        current_globally_loaded_model_value: Optional[str], load_4bit: bool):
    if source_dropdown_val != "Local (Oobabooga)":
        yield f"**Status:** Ready. Current local: `{current_globally_loaded_model_value or 'None'}`.", current_globally_loaded_model_value;
        return
    if not selected_model_in_dropdown:
        yield f"**Status:** No local model selected. Current: `{current_globally_loaded_model_value or 'None'}`.", current_globally_loaded_model_value;
        return
    if selected_model_in_dropdown == current_globally_loaded_model_value:
        yield f"**Status:** Model `{selected_model_in_dropdown}` is already loaded.", current_globally_loaded_model_value;
        return
    unloaded_prev_model_msg = ""
    model_at_start_of_operation = current_globally_loaded_model_value
    if current_globally_loaded_model_value and current_globally_loaded_model_value != selected_model_in_dropdown:
        yield f"**Status:** Unloading `{current_globally_loaded_model_value}`...", current_globally_loaded_model_value
        unloaded_prev_model_msg = unload_local_model(current_globally_loaded_model_value)
        if "unloaded" in unloaded_prev_model_msg.lower() or "successfully" in unloaded_prev_model_msg.lower():
            model_at_start_of_operation = None
    yield f"**Status:** ⌛ Loading `{selected_model_in_dropdown}` (4-bit: {load_4bit})...", model_at_start_of_operation
    status_msg, new_model_name_if_loaded = load_local_model(selected_model_in_dropdown, load_4bit)
    final_model_state_to_yield = None
    if new_model_name_if_loaded:
        final_model_state_to_yield = new_model_name_if_loaded
    elif "unloaded" in (unloaded_prev_model_msg or "").lower() or (
            "successfully" in (unloaded_prev_model_msg or "").lower() and not new_model_name_if_loaded):
        final_model_state_to_yield = None
    else:
        final_model_state_to_yield = model_at_start_of_operation
    yield f"**Status:** {status_msg}", final_model_state_to_yield


with gr.Blocks(theme=gr.themes.Soft(), title="Advanced Chat Playground with Judging") as demo:
    current_loaded_local_model = gr.State(None)
    last_user_prompt_for_judge = gr.State("")

    gr.Markdown("# Advanced Chat Playground & LLM Judge")
    gr.Markdown(
        "Compare chat models side-by-side from multiple providers, including OpenRouter, Google Gemini, and local Oobabooga instances. Use the integrated LLM Judge to rank responses and build a DPO dataset.")
    loading_status = gr.Markdown("**Status:** Ready.")
    with gr.Row():
        (source_a, or_key_a, gemini_key_a, or_group_a, gemini_group_a, model_a, max_tokens_slider_a,
         chatbot_a) = create_chat_column("Chatbot A")
        (source_b, or_key_b, gemini_key_b, or_group_b, gemini_group_b, model_b, max_tokens_slider_b,
         chatbot_b) = create_chat_column("Chatbot B")
    with gr.Accordion("⚙️ Shared Generation Parameters", open=False):
        with gr.Row(): load_4bit_checkbox = gr.Checkbox(label="Load Local Model in 4-bit", value=True)
        with gr.Row():
            temperature_slider = gr.Slider(0.0, 2.0, 0.7, step=0.1, label="Temperature")
            top_p_slider = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="Top-P")

    with gr.Row():
        load_manual_prompt_dd = gr.Dropdown(label="Saved Prompts", scale=2, container=False, interactive=True)
        user_input = gr.Textbox(placeholder="Type prompt here for manual generation or select a saved one...", scale=7,
                                show_label=False, container=False)
        save_manual_prompt_btn = gr.Button("💾 Save", scale=1, min_width=80)

    with gr.Row(equal_height=True):
        send_button = gr.Button("Send", variant="primary", scale=1, min_width=100)
        stop_button = gr.Button("Stop Gen", variant="stop", scale=1, min_width=100)
        clear_button = gr.Button("🗑️ Clear Chats", scale=1, min_width=100)

    with gr.Accordion("⚖️ Judge Configuration & Actions", open=True):
        judge_loop_status_display = gr.Textbox(label="Judge/Loop Status", interactive=False, lines=1, value="Ready.")
        with gr.Row():
            with gr.Column(scale=1):
                judge_type_radio = gr.Radio(
                    ["OpenAI / Cloud API", "OpenRouter", "Google Gemini", "LM Studio", "Oobabooga (Local Judge)"],
                    label="Judge Type",
                    value="OpenAI / Cloud API")
                ooba_judge_note = gr.Markdown("**Ooba Judge**: Uses *currently loaded* Ooba model.", visible=False)
                with gr.Group(visible=True) as judge_openai_group:
                    judge_openai_api_url_input = gr.Textbox(label="Judge API URL (OpenAI)",
                                                            value=DEFAULT_OPENAI_API_URL)
                    judge_openai_api_key_input = gr.Textbox(label="Judge API Key", type="password")
                    with gr.Row(): judge_openai_model_dropdown = gr.Dropdown(label="Judge Model", interactive=True,
                                                                             allow_custom_value=True); judge_refresh_openai_btn = gr.Button(
                        "🔄")
                with gr.Group(visible=False) as judge_or_group:
                    judge_or_api_url_input = gr.Textbox(label="Judge API URL (OpenRouter)",
                                                        value=DEFAULT_OPENROUTER_API_URL)
                    judge_or_api_key_input = gr.Textbox(label="OR API Key (Judge)", type="password")
                    with gr.Row(): judge_or_model_dropdown = gr.Dropdown(label="Judge Model", interactive=True,
                                                                         allow_custom_value=True); judge_refresh_or_btn = gr.Button(
                        "🔄")
                    with gr.Accordion("Opt. OR Headers (Judge)", open=False): judge_or_http_referer_input = gr.Textbox(
                        label="HTTP-Referer"); judge_or_x_title_input = gr.Textbox(label="X-Title")
                with gr.Group(visible=False) as judge_gemini_group:
                    judge_gemini_api_key_input = gr.Textbox(label="Google AI Studio API Key (Judge)", type="password")
                    judge_gemini_model_dropdown = gr.Dropdown(label="Judge Model (Gemini)", choices=get_gemini_models(),
                                                              value="gemini-1.5-flash-latest", interactive=True,
                                                              allow_custom_value=True)
                with gr.Group(visible=False) as judge_lmstudio_group:
                    judge_lmstudio_api_url_input = gr.Textbox(label="LM Studio API URL", value=DEFAULT_LMSTUDIO_API_URL)
                    judge_lmstudio_model_input = gr.Textbox(label="Model ID (LM Studio)",
                                                            info="Uses model loaded in LM Studio UI.")
                    judge_test_lmstudio_btn = gr.Button("🔌 Test LM Studio")

            with gr.Column(scale=1):
                judge_rationale_textbox = gr.Textbox(label="🧑‍⚖️ Rationale / Override", lines=3, interactive=True)
                preliminary_judge_choice = gr.State("Error")
                gr.Markdown("💾 **Manual Save (uses current chats):**");
                with gr.Row(): save_a_better_btn = gr.Button("A Wins 👍"); save_b_better_btn = gr.Button(
                    "B Wins 👍"); save_both_bad_btn = gr.Button("Both Bad 👎"); auto_judge_btn = gr.Button("🤖 Auto Judge",
                                                                                                         variant="secondary")
                gr.Markdown("--- \n🔄 **Auto Judge Loop (uses a dedicated prompt below):**")
                with gr.Row():
                    load_loop_prompt_dd = gr.Dropdown(label="Saved Loop Prompts", scale=2, container=False,
                                                      interactive=True)
                    judge_loop_prompt_input = gr.Textbox(label="Auto-Judge Loop Prompt", scale=7, container=False,
                                                         info="This prompt will be used for each cycle.")
                    save_loop_prompt_btn = gr.Button("💾 Save", scale=1, min_width=80)
                judge_loop_delay_slider = gr.Slider(5, 120, 10, step=5, label="Delay Between Cycles (s)")
                with gr.Row(): start_judge_loop_btn = gr.Button("▶️ Start Loop",
                                                                variant="primary"); stop_judge_loop_btn = gr.Button(
                    "⏹️ Stop Loop", variant="stop")


        def switch_judge_ui_visibility(judge_type_val):
            return {
                judge_openai_group: gr.update(visible=judge_type_val == "OpenAI / Cloud API"),
                judge_or_group: gr.update(visible=judge_type_val == "OpenRouter"),
                judge_gemini_group: gr.update(visible=judge_type_val == "Google Gemini"),
                judge_lmstudio_group: gr.update(visible=judge_type_val == "LM Studio"),
                ooba_judge_note: gr.update(visible=judge_type_val == "Oobabooga (Local Judge)")
            }


        judge_type_radio.change(switch_judge_ui_visibility, judge_type_radio,
                                [judge_openai_group, judge_or_group, judge_gemini_group, judge_lmstudio_group,
                                 ooba_judge_note])

    with gr.Tab("📖 Judgement Log"):
        refresh_log_btn = gr.Button("🔄 Refresh Log")
        judge_log_display = gr.DataFrame(headers=["Timestamp", "Winner", "Prompt Snippet", "Rationale Snippet"],
                                         datatype=["str"] * 4, interactive=False, wrap=True)

    # --- Event Listeners ---
    source_a.change(update_model_list, [source_a, or_key_a, gemini_key_a], [model_a, or_group_a, gemini_group_a])
    or_key_a.input(update_model_list, [source_a, or_key_a, gemini_key_a], [model_a, or_group_a, gemini_group_a],
                   show_progress="hidden")
    gemini_key_a.input(update_model_list, [source_a, or_key_a, gemini_key_a], [model_a, or_group_a, gemini_group_a],
                       show_progress="hidden")

    source_b.change(update_model_list, [source_b, or_key_b, gemini_key_b], [model_b, or_group_b, gemini_group_b])
    or_key_b.input(update_model_list, [source_b, or_key_b, gemini_key_b], [model_b, or_group_b, gemini_group_b],
                   show_progress="hidden")
    gemini_key_b.input(update_model_list, [source_b, or_key_b, gemini_key_b], [model_b, or_group_b, gemini_group_b],
                       show_progress="hidden")

    model_a.change(handle_local_model_selection_change,
                   [source_a, model_a, current_loaded_local_model, load_4bit_checkbox],
                   [loading_status, current_loaded_local_model])
    model_b.change(handle_local_model_selection_change,
                   [source_b, model_b, current_loaded_local_model, load_4bit_checkbox],
                   [loading_status, current_loaded_local_model])

    chat_stream_inputs = [user_input,
                          chatbot_a, source_a, or_key_a, gemini_key_a, model_a, max_tokens_slider_a,
                          chatbot_b, source_b, or_key_b, gemini_key_b, model_b, max_tokens_slider_b,
                          temperature_slider, top_p_slider, current_loaded_local_model]
    chat_stream_outputs = [chatbot_a, chatbot_b, last_user_prompt_for_judge, chatbot_a, chatbot_b]
    gen_event = send_button.click(run_chat_streams, chat_stream_inputs, chat_stream_outputs, concurrency_limit=1,
                                  show_progress="full")
    submit_event = user_input.submit(run_chat_streams, chat_stream_inputs, chat_stream_outputs, concurrency_limit=1,
                                     show_progress="full")
    gen_event.then(lambda: gr.update(value=""), inputs=None, outputs=user_input, js="() => {return ''}")
    submit_event.then(lambda: gr.update(value=""), inputs=None, outputs=user_input, js="() => {return ''}")
    stop_button.click(fn=None, inputs=None, outputs=None, cancels=[gen_event, submit_event], queue=False,
                      show_progress="hidden")
    clear_button.click(lambda: ([], [], "", [], []), None,
                       [chatbot_a, chatbot_b, user_input, last_user_prompt_for_judge], queue=False,
                       show_progress="hidden")

    judge_refresh_openai_btn.click(list_api_models_for_judge, [judge_openai_api_url_input, judge_openai_api_key_input],
                                   [judge_openai_model_dropdown, judge_loop_status_display], show_progress="hidden")
    judge_refresh_or_btn.click(list_api_models_for_judge, [judge_or_api_url_input, judge_or_api_key_input],
                               [judge_or_model_dropdown, judge_loop_status_display], show_progress="hidden")
    judge_test_lmstudio_btn.click(list_api_models_for_judge, [judge_lmstudio_api_url_input, gr.State("NA")],
                                  [gr.State(None), judge_loop_status_display], show_progress="hidden")


    def save_manual_judgement_wrapper(choice_btn, user_prompt_from_state, current_hist_a, current_hist_b,
                                      rationale_text):
        prompt_to_save = user_prompt_from_state
        if not prompt_to_save:
            user_turns_a = [msg['content'] for msg in current_hist_a if msg['role'] == 'user']
            user_turns_b = [msg['content'] for msg in current_hist_b if msg['role'] == 'user']
            if user_turns_a:
                prompt_to_save = user_turns_a[-1]
            elif user_turns_b:
                prompt_to_save = user_turns_b[-1]
            else:
                prompt_to_save = "User prompt not captured for saving."
        resp_a_text = current_hist_a[-1]['content'] if current_hist_a and current_hist_a[-1][
            'role'] == 'assistant' else "(No A response found in chat)"
        resp_b_text = current_hist_b[-1]['content'] if current_hist_b and current_hist_b[-1][
            'role'] == 'assistant' else "(No B response found in chat)"
        status_msg, success = save_preference(prompt_to_save, resp_a_text, resp_b_text, choice_btn, rationale_text,
                                              DPO_FILE, LOG_FILE)
        return status_msg, load_and_format_log(LOG_FILE)


    save_a_better_btn.click(partial(save_manual_judgement_wrapper, "A is better"),
                            [last_user_prompt_for_judge, chatbot_a, chatbot_b, judge_rationale_textbox],
                            [judge_loop_status_display, judge_log_display], show_progress="hidden")
    save_b_better_btn.click(partial(save_manual_judgement_wrapper, "B is better"),
                            [last_user_prompt_for_judge, chatbot_a, chatbot_b, judge_rationale_textbox],
                            [judge_loop_status_display, judge_log_display], show_progress="hidden")
    save_both_bad_btn.click(partial(save_manual_judgement_wrapper, "Both are bad / Tie"),
                            [last_user_prompt_for_judge, chatbot_a, chatbot_b, judge_rationale_textbox],
                            [judge_loop_status_display, judge_log_display], show_progress="hidden")

    # Auto Judge Button Inputs
    auto_judge_inputs = [last_user_prompt_for_judge, chatbot_a, chatbot_b, judge_type_radio, current_loaded_local_model,
                         judge_openai_api_url_input, judge_openai_api_key_input, judge_openai_model_dropdown,
                         judge_or_api_url_input, judge_or_api_key_input, judge_or_model_dropdown,
                         judge_or_http_referer_input, judge_or_x_title_input,
                         judge_lmstudio_api_url_input, judge_lmstudio_model_input,
                         judge_gemini_api_key_input, judge_gemini_model_dropdown]
    auto_judge_outputs = [judge_loop_status_display, judge_rationale_textbox, preliminary_judge_choice,
                          judge_log_display]
    auto_judge_btn.click(call_auto_judge_interface, auto_judge_inputs, auto_judge_outputs)

    refresh_log_btn.click(lambda: load_and_format_log(LOG_FILE), [], judge_log_display, show_progress="hidden")

    loop_generation_params = [source_a, or_key_a, gemini_key_a, model_a, max_tokens_slider_a, source_b, or_key_b,
                              gemini_key_b, model_b, max_tokens_slider_b, temperature_slider, top_p_slider,
                              current_loaded_local_model]
    loop_judge_config_params = [judge_type_radio, judge_openai_api_url_input, judge_openai_api_key_input,
                                judge_openai_model_dropdown,
                                judge_or_api_url_input, judge_or_api_key_input, judge_or_model_dropdown,
                                judge_or_http_referer_input, judge_or_x_title_input,
                                judge_lmstudio_api_url_input, judge_lmstudio_model_input,
                                judge_gemini_api_key_input, judge_gemini_model_dropdown]
    loop_control_params = [judge_loop_delay_slider]
    all_judge_loop_inputs = [judge_loop_prompt_input] + loop_generation_params + loop_judge_config_params + loop_control_params
    judge_loop_outputs = [judge_loop_status_display, chatbot_a, chatbot_b, judge_rationale_textbox, gr.State(None),
                          gr.State(None), gr.State(None), gr.State(None), preliminary_judge_choice, judge_log_display]
    loop_event = start_judge_loop_btn.click(auto_judge_loop_runner, inputs=all_judge_loop_inputs,
                                            outputs=judge_loop_outputs, concurrency_limit=1)
    stop_judge_loop_btn.click(stop_judge_loop_global, None, judge_loop_status_display, cancels=[loop_event],
                              queue=False, show_progress="hidden")

    # --- Settings and Prompt Listeners ---
    api_key_inputs = [or_key_a, or_key_b, gemini_key_a, gemini_key_b, judge_openai_api_key_input,
                      judge_or_api_key_input, judge_gemini_api_key_input]
    for key_input in api_key_inputs:
        key_input.blur(save_api_keys, inputs=api_key_inputs, outputs=None, show_progress="hidden")

    save_manual_prompt_btn.click(save_new_prompt, inputs=[gr.State("manual"), user_input],
                                 outputs=[load_manual_prompt_dd], show_progress="hidden")
    save_loop_prompt_btn.click(save_new_prompt, inputs=[gr.State("loop"), judge_loop_prompt_input],
                               outputs=[load_loop_prompt_dd], show_progress="hidden")
    load_manual_prompt_dd.change(lambda x: x, inputs=load_manual_prompt_dd, outputs=user_input, show_progress="hidden")
    load_loop_prompt_dd.change(lambda x: x, inputs=load_loop_prompt_dd, outputs=judge_loop_prompt_input,
                               show_progress="hidden")


    def initial_load_and_setup():
        settings = load_settings()
        api_keys = settings.get("api_keys", {})
        prompts = settings.get("prompts", {})
        log_data = load_and_format_log(LOG_FILE)
        default_judge_type = "OpenAI / Cloud API"
        judge_ui_updates = switch_judge_ui_visibility(default_judge_type)

        # We need to run async functions in a new event loop for initial setup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        model_list_a_update, or_group_a_vis, gem_group_a_vis = loop.run_until_complete(
            update_model_list("OpenRouter", api_keys.get("or_key_a", ""), api_keys.get("gemini_key_a", "")))
        model_list_b_update, or_group_b_vis, gem_group_b_vis = loop.run_until_complete(
            update_model_list("OpenRouter", api_keys.get("or_key_b", ""), api_keys.get("gemini_key_b", "")))

        loop.close()

        return (
            log_data, judge_ui_updates[judge_openai_group], judge_ui_updates[judge_or_group],
            judge_ui_updates[judge_gemini_group],
            judge_ui_updates[judge_lmstudio_group], judge_ui_updates[ooba_judge_note],
            model_list_a_update, or_group_a_vis, gem_group_a_vis,
            model_list_b_update, or_group_b_vis, gem_group_b_vis,
            api_keys.get("or_key_a", ""), api_keys.get("or_key_b", ""),
            api_keys.get("gemini_key_a", ""), api_keys.get("gemini_key_b", ""),
            api_keys.get("judge_openai_api_key", ""), api_keys.get("judge_or_api_key", ""),
            api_keys.get("judge_gemini_api_key", ""),
            gr.update(choices=prompts.get("manual", [])), gr.update(choices=prompts.get("loop", []))
        )


    load_outputs = [
        judge_log_display, judge_openai_group, judge_or_group, judge_gemini_group, judge_lmstudio_group,
        ooba_judge_note,
        model_a, or_group_a, gemini_group_a,
        model_b, or_group_b, gemini_group_b,
        or_key_a, or_key_b, gemini_key_a, gemini_key_b,
        judge_openai_api_key_input, judge_or_api_key_input, judge_gemini_api_key_input,
        load_manual_prompt_dd, load_loop_prompt_dd
    ]
    demo.load(initial_load_and_setup, None, load_outputs)

if __name__ == "__main__":
    demo.queue().launch(debug=True)