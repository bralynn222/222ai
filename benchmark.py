import gradio as gr
import json
import os
import time
from datetime import datetime
import asyncio
from openai import AsyncOpenAI, APIError, AuthenticationError

# --- Configuration ---
# Default URLs from the provided reference code
DEFAULT_OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_API_URL = "https://api.openai.com/v1"
DEFAULT_OOBABOOGA_API_URL = "http://127.0.0.1:5000/v1"
DEFAULT_LMSTUDIO_API_URL = "http://127.0.0.1:1234/v1"

# --- Create an example file for Gradio Examples ---
try:
    examples_dir = "example_files"
    if not os.path.exists(examples_dir):
        os.makedirs(examples_dir)
    example_file_path = os.path.join(examples_dir, "prompts.json")
    example_content = [
        "Create a Python function that sorts a list of tuples based on the second element in each tuple.",
        "Write a brief explanation of how asynchronous programming works in Python using asyncio.",
        "Generate a Python script to fetch and display the current weather for a given city using a free weather API."
    ]
    with open(example_file_path, 'w', encoding='utf-8') as f:
        json.dump(example_content, f, indent=2)
except Exception as e:
    print(f"Warning: Could not create example file. gr.Examples might not work. Error: {e}")
    example_file_path = None


# --- API Helper Functions (Adapted from your provided code) ---

async def create_async_client(api_service: str, api_url: str, api_key: str):
    """Creates an Asynchronous OpenAI client configured for the selected service."""
    if not api_url:
        raise ValueError("API URL cannot be empty.")

    headers = {}
    if api_service == "OpenRouter":
        headers = {
            "HTTP-Referer": "http://localhost:7860",  # Placeholder
            "X-Title": "Gradio Batch Processor",  # Placeholder
        }

    return AsyncOpenAI(
        api_key=api_key or "NA",  # Use "NA" for local models if no key is needed
        base_url=api_url,
        default_headers=headers
    )


async def fetch_models(api_service: str, api_url: str, api_key: str):
    """Fetches available models from the selected API service."""
    if not api_url:
        gr.Warning("API URL is missing.")
        return gr.Dropdown(choices=[], value=None, interactive=False)
    if api_service in ["OpenAI", "OpenRouter"] and not api_key:
        gr.Warning(f"API Key is required for {api_service}.")
        return gr.Dropdown(choices=[], value=None, interactive=False)

    try:
        client = await create_async_client(api_service, api_url, api_key)
        models_response = await client.models.list()
        all_models = sorted([model.id for model in models_response.data])

        gr.Info(f"Successfully loaded {len(all_models)} models from {api_service}.")

        if not all_models:
            gr.Warning("No models found. Check your API key, URL, and service status.")
            return gr.Dropdown(choices=[], value=None, interactive=False)

        default_model = None
        # Try to find a sensible default
        preferred_defaults = ["openai/gpt-4o", "gpt-4o", "gpt-3.5-turbo"]
        for m in preferred_defaults:
            if m in all_models:
                default_model = m
                break
        if not default_model:
            default_model = all_models[0]

        return gr.Dropdown(choices=all_models, value=default_model, interactive=True)

    except AuthenticationError:
        gr.Error(f"Authentication Error: The provided API Key for {api_service} is invalid.")
        return gr.Dropdown(choices=[], value=None, interactive=False)
    except APIError as e:
        gr.Error(f"API Error fetching models from {api_service}: {e.message}")
        return gr.Dropdown(choices=[], value=None, interactive=False)
    except Exception as e:
        gr.Error(f"An error occurred while fetching models: {e}")
        return gr.Dropdown(choices=[], value=None, interactive=False)


async def process_prompts(api_service, api_url, api_key, model, json_file, delay_seconds,
                          progress=gr.Progress(track_tqdm=True)):
    """
    Processes prompts from a JSON file using the selected API service.
    """
    # Validation checks
    if not api_service: yield "Error: Please select an API service.", gr.Button(visible=False); return
    if not api_url: yield f"Error: API URL for {api_service} is required.", gr.Button(visible=False); return
    if api_service in ["OpenAI", "OpenRouter"] and not api_key:
        yield f"Error: API Key for {api_service} is required.", gr.Button(visible=False);
        return
    if not model: yield "Error: Please select a model.", gr.Button(visible=False); return
    if json_file is None: yield "Error: Please upload a JSON file.", gr.Button(visible=False); return

    try:
        client = await create_async_client(api_service, api_url, api_key)
        with open(json_file.name, 'r', encoding='utf-8') as f:
            prompts = json.load(f)
        if not isinstance(prompts, list):
            yield "Error: The JSON file must contain a list of strings.", gr.Button(visible=False);
            return
    except (json.JSONDecodeError, ValueError) as e:
        yield f"Error processing setup: {e}", gr.Button(visible=False);
        return
    except Exception as e:
        yield f"An unexpected error occurred during setup: {e}", gr.Button(visible=False);
        return

    full_output = ""
    num_prompts = len(prompts)

    for i, prompt in enumerate(progress.tqdm(prompts, desc="Processing Prompts")):
        status_md = f"## ➡️ Prompt {i + 1}/{num_prompts}\n\n**User Request:**\n```\n{prompt}\n```\n\n"
        full_output += status_md
        yield full_output + "\n*Waiting for AI response...*", gr.Button(visible=False)

        try:
            stream = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                stream=True
            )

            ai_response = ""
            full_output += f"**🤖 AI Response ({model}):**\n```\n"
            async for chunk in stream:
                token = chunk.choices[0].delta.content
                if token:
                    ai_response += token
                    yield full_output + ai_response + "...`", gr.Button(visible=False)
                    await asyncio.sleep(0.001)

            full_output += ai_response + "\n```\n\n---\n\n"
            yield full_output, gr.Button(visible=False)

        except APIError as e:
            error_message = f"API Error on prompt {i + 1}: {e.message}"
            full_output += f"**❌ Error:**\n```\n{error_message}\n```\n\n---\n\n"
            yield full_output, gr.Button(visible=False)
        except Exception as e:
            error_message = f"An unexpected error occurred on prompt {i + 1}: {e}"
            full_output += f"**❌ Error:**\n```\n{error_message}\n```\n\n---\n\n"
            yield full_output, gr.Button(visible=False)

        if i < num_prompts - 1 and delay_seconds > 0:
            yield full_output + f"\n*Pausing for {delay_seconds} second(s)...*", gr.Button(visible=False)
            await asyncio.sleep(delay_seconds)

    yield full_output, gr.Button(visible=True)  # Show save button when done


def save_results_to_file(markdown_content):
    if not markdown_content or not markdown_content.strip():
        gr.Warning("There is no content to save.")
        return None
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"batch_results_{timestamp}.md"
    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        gr.Info(f"Results saved to {filename}")
        return gr.File(value=filename, visible=True)
    except Exception as e:
        gr.Error(f"Failed to save file: {e}");
        return None


# --- Gradio UI ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Universal Batch Prompt Processor")
    gr.Markdown("Select an API service, provide credentials, upload a JSON file of prompts, and run.")

    # --- API Configuration Section ---
    with gr.Accordion("1. API Configuration", open=True):
        api_service_radio = gr.Radio(
            ["OpenRouter", "OpenAI", "Oobabooga (Local)", "LM Studio"],
            label="Select API Service",
            value="OpenRouter"
        )
        # Group for OpenRouter
        with gr.Group(visible=True) as or_group:
            or_api_url = gr.Textbox(label="OpenRouter API URL", value=DEFAULT_OPENROUTER_API_URL)
            or_api_key = gr.Textbox(label="OpenRouter API Key", type="password", placeholder="Enter sk-or-...")
        # Group for OpenAI
        with gr.Group(visible=False) as oai_group:
            oai_api_url = gr.Textbox(label="OpenAI API URL", value=DEFAULT_OPENAI_API_URL)
            oai_api_key = gr.Textbox(label="OpenAI API Key", type="password", placeholder="Enter sk-...")
        # Group for Oobabooga
        with gr.Group(visible=False) as ooba_group:
            ooba_api_url = gr.Textbox(label="Oobabooga API URL", value=DEFAULT_OOBABOOGA_API_URL)
            ooba_api_key = gr.Textbox(label="API Key (if required)", type="password")
        # Group for LM Studio
        with gr.Group(visible=False) as lms_group:
            lms_api_url = gr.Textbox(label="LM Studio API URL", value=DEFAULT_LMSTUDIO_API_URL)
            lms_api_key = gr.Textbox(label="API Key (if required)", type="password")

        load_models_button = gr.Button("Load Models from Selected Service")

    # --- Prompt & Execution Section ---
    with gr.Accordion("2. Prompt and Execution", open=True):
        model_dropdown = gr.Dropdown(label="Select Model", interactive=False, allow_custom_value=True)
        json_uploader = gr.File(label='Upload JSON Prompts File (e.g., ["prompt 1", ...])', file_types=[".json"])
        delay_slider = gr.Slider(minimum=0, maximum=60, step=1, value=2, label="Delay Between Prompts (seconds)")
        with gr.Row():
            run_button = gr.Button("Run All Prompts", variant="primary")
            stop_button = gr.Button("Stop", variant="stop", visible=False)

    # --- Results Section ---
    with gr.Accordion("3. Results", open=True):
        output_markdown = gr.Markdown(label="Live Output")
        save_button = gr.Button("Download Results to File", visible=False)
        download_file_output = gr.File(label="Download", visible=False)

    # --- Examples Section ---
    if example_file_path:
        gr.Examples(examples=[[example_file_path]], inputs=[json_uploader], label="Clickable Example")

    # --- State Management (to pass correct URL/Key to functions) ---
    current_api_url = gr.State("")
    current_api_key = gr.State("")


    # --- Event Handlers ---
    def switch_api_visibility(service):
        """Updates UI visibility and sets the current URL/Key for other functions to use."""
        url_val, key_val = "", ""
        if service == "OpenRouter":
            url_val, key_val = or_api_url.value, or_api_key.value
        elif service == "OpenAI":
            url_val, key_val = oai_api_url.value, oai_api_key.value
        elif service == "Oobabooga (Local)":
            url_val, key_val = ooba_api_url.value, ooba_api_key.value
        elif service == "LM Studio":
            url_val, key_val = lms_api_url.value, lms_api_key.value

        return {
            or_group: gr.update(visible=service == "OpenRouter"),
            oai_group: gr.update(visible=service == "OpenAI"),
            ooba_group: gr.update(visible=service == "Oobabooga (Local)"),
            lms_group: gr.update(visible=service == "LM Studio"),
            current_api_url: url_val,
            current_api_key: key_val,
            model_dropdown: gr.Dropdown(choices=[], value=None, label="Select Model", interactive=False)
            # Reset model list
        }


    api_service_radio.change(
        fn=switch_api_visibility,
        inputs=[api_service_radio],
        outputs=[or_group, oai_group, ooba_group, lms_group, current_api_url, current_api_key, model_dropdown]
    )


    # Function to grab the latest values from the visible textboxes before an action
    def get_current_api_config(service, or_url, or_key, oai_url, oai_key, ooba_url, ooba_key, lms_url, lms_key):
        if service == "OpenRouter": return or_url, or_key
        if service == "OpenAI": return oai_url, oai_key
        if service == "Oobabooga (Local)": return ooba_url, ooba_key
        if service == "LM Studio": return lms_url, lms_key
        return "", ""


    api_config_inputs = [api_service_radio, or_api_url, or_api_key, oai_api_url, oai_api_key, ooba_api_url,
                         ooba_api_key, lms_api_url, lms_api_key]

    load_models_button.click(
        fn=get_current_api_config, inputs=api_config_inputs, outputs=[current_api_url, current_api_key]
    ).then(
        fn=fetch_models, inputs=[api_service_radio, current_api_url, current_api_key], outputs=[model_dropdown]
    )

    run_event = run_button.click(
        fn=lambda: (gr.Button(visible=False), gr.Button(visible=True)), outputs=[run_button, stop_button]
    ).then(
        fn=get_current_api_config, inputs=api_config_inputs, outputs=[current_api_url, current_api_key]
    ).then(
        fn=process_prompts,
        inputs=[api_service_radio, current_api_url, current_api_key, model_dropdown, json_uploader, delay_slider],
        outputs=[output_markdown, save_button]
    ).then(
        fn=lambda: (gr.Button(visible=True), gr.Button(visible=False)), outputs=[run_button, stop_button]
    )

    stop_button.click(
        fn=lambda: (gr.Button(visible=True), gr.Button(visible=False)),
        outputs=[run_button, stop_button], cancels=[run_event], queue=False
    )

    save_button.click(fn=save_results_to_file, inputs=[output_markdown], outputs=[download_file_output])

    # Initialize the UI on first load
    demo.load(
        fn=switch_api_visibility,
        inputs=[api_service_radio],
        outputs=[or_group, oai_group, ooba_group, lms_group, current_api_url, current_api_key, model_dropdown]
    )

if __name__ == "__main__":
    demo.queue().launch()