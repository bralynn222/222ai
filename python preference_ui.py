import gradio as gr
import requests
import json
import os
from typing import Optional

# --- Global Configuration ---
DATA_FILE = "preferences.jsonl"
STOP_GENERATION = False


# --- Core Functions ---
def stop_inference():
    """Sets a global flag to stop the generation loops."""
    global STOP_GENERATION
    STOP_GENERATION = True
    print("Stop signal received.")
    return "Stopping generation..."


def get_api_response(api_url: str, prompt: str, max_tokens: Optional[int], temp: Optional[float],
                     top_p: Optional[float]):
    """
    Helper function to call the OpenAI-compatible streaming API.
    It conditionally adds parameters to the request.
    """
    global STOP_GENERATION

    if not api_url.endswith("/"):
        api_url += "/"
    stream_endpoint = api_url + "v1/chat/completions"

    request_payload = {
        "model": "loaded_model",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        'stream': True,
    }

    if max_tokens is not None:
        request_payload['max_tokens'] = max_tokens
    if temp is not None:
        request_payload['temperature'] = temp
    if top_p is not None:
        request_payload['top_p'] = top_p

    try:
        response = requests.post(stream_endpoint, json=request_payload, stream=True)
        response.raise_for_status()

        for line in response.iter_lines():
            if STOP_GENERATION:
                print("Breaking generation loop.")
                break
            if line:
                decoded_line = line.decode('utf-8')
                if decoded_line.startswith('data:'):
                    if '[DONE]' in decoded_line:
                        break
                    try:
                        json_data = json.loads(decoded_line[len('data: '):])
                        delta = json_data.get('choices', [{}])[0].get('delta', {})
                        new_text = delta.get('content', '')
                        if new_text:
                            yield new_text
                    except (json.JSONDecodeError, IndexError):
                        continue

    except requests.exceptions.RequestException as e:
        print(f"API Error: {e}")
        yield f"[API ERROR: {e}]"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        yield f"[UNEXPECTED ERROR: {e}]"


# MODIFIED: Added `manual_b_enabled` boolean to the signature.
def generate_stream(api_url: str, prompt: str, override_params: bool, max_tokens: int, temp_a: float, top_p_a: float,
                    temp_b: float, top_p_b: float, manual_b_enabled: bool):
    """
    Main generator function. Skips generating Response B if manual_b_enabled is True.
    """
    global STOP_GENERATION
    STOP_GENERATION = False

    if not api_url:
        yield "API URL cannot be empty.", "", "Error: Please enter the API URL.", prompt, "", ""
        return

    if not override_params:
        max_tokens = None
        temp_a, top_p_a = None, None
        temp_b, top_p_b = None, None
    elif temp_a <= 0 or temp_b <= 0:
        yield "", "", "Error: If overriding, Temperature must be > 0.", prompt, "", ""
        return

    # --- Generate Response A ---
    response_a_text = ""
    yield "", "", "Generating Response A...", prompt, "", ""
    for new_text in get_api_response(api_url, prompt, max_tokens, temp_a, top_p_a):
        if STOP_GENERATION: break
        response_a_text += new_text
        yield response_a_text, "", "Generating Response A...", prompt, "", ""
    if STOP_GENERATION:
        yield response_a_text, "", "Stopped.", prompt, response_a_text, ""
        return

    # MODIFIED: Conditionally handle Response B
    if manual_b_enabled:
        # If manual input is enabled, we stop here and let the user type.
        yield response_a_text, "", "Generation A complete. Enter Response B manually.", prompt, response_a_text, ""
        return

    # --- Generate Response B (only if not manual) ---
    response_b_text = ""
    yield response_a_text, "", "Generating Response B...", prompt, response_a_text, ""
    for new_text in get_api_response(api_url, prompt, max_tokens, temp_b, top_p_b):
        if STOP_GENERATION: break
        response_b_text += new_text
        yield response_a_text, response_b_text, "Generating Response B...", prompt, response_a_text, ""
    if STOP_GENERATION:
        yield response_a_text, response_b_text, "Stopped.", prompt, response_a_text, response_b_text
        return

    yield response_a_text, response_b_text, "Complete!", prompt, response_a_text, response_b_text


def save_preference(prompt: str, response_a: str, response_b: str, choice: str, comment: str):
    """Saves the user's preference to the JSONL file."""
    if not prompt or not response_a or not response_b:
        return "Cannot save. One of the required fields (Prompt, Response A, Response B) is empty.", "", "", ""

    if "API ERROR" in response_a or "API ERROR" in response_b:
        return "Cannot save. An API error was present in the generation.", "", "", ""

    if choice == "A is better":
        chosen, rejected = response_a, response_b
    elif choice == "B is better":
        chosen, rejected = response_b, response_a
    elif choice == "Both are bad":
        chosen, rejected = "N/A (Both Bad)", f"A: {response_a}\n---\nB: {response_b}"
    else:
        # This case should not be reached with the current UI
        return "Not saved (Invalid Choice).", "", "", ""

    data_record = {"prompt": prompt, "chosen": chosen, "rejected": rejected, "comment": comment or "N/A"}
    with open(DATA_FILE, 'a') as f:
        f.write(json.dumps(data_record) + "\n")

    # Clear the textboxes and comment field after successful save
    return f"Saved preference to {DATA_FILE}!", "", "", ""


def build_ui():
    """Builds the Gradio UI application."""
    if not os.path.exists(DATA_FILE):
        open(DATA_FILE, 'w').close()

    with gr.Blocks(theme=gr.themes.Default(), title="DPO Data Collector") as demo:
        gr.Markdown("# 🤖 LLM Preference Collector (via API) By Bralynn Matthew Tipps")
        gr.Markdown(
            "This UI connects to a running `text-generation-webui` backend. Launch the backend with the `--api` flag and load your model there first.")

        # State variables to hold the "official" generated data
        stored_prompt = gr.State("")
        stored_response_a = gr.State("")
        stored_response_b = gr.State("") # This will be empty in manual mode

        with gr.Row():
            api_url_input = gr.Textbox(
                label="Text Generation WebUI API URL",
                value="http://127.0.0.1:5000",
                info="The URL of your oobabooga API server (OpenAI-compatible)."
            )
            status_display = gr.Textbox(
                label="Status",
                interactive=False,
                value="Ready. Enter a prompt and generate."
            )

        override_params_checkbox = gr.Checkbox(
            label="Override server default parameters",
            value=False,
            info="Check this to manually set generation parameters below. Otherwise, server defaults will be used."
        )

        with gr.Accordion("Generation Parameters", open=False, visible=False) as params_accordion:
            max_tokens_slider = gr.Slider(minimum=256, maximum=16384, value=4096, step=256, label="Max New Tokens")
            with gr.Row():
                temp_a_slider = gr.Slider(minimum=0.01, maximum=2.0, value=0.6, step=0.05, label="Temperature (A)")
                top_p_a_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top-P (A)")
            with gr.Row():
                temp_b_slider = gr.Slider(minimum=0.01, maximum=2.0, value=0.8, step=0.05, label="Temperature (B)")
                top_p_b_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.9, step=0.05, label="Top-P (B)")

        override_params_checkbox.change(
            fn=lambda x: gr.update(visible=x, open=x),
            inputs=override_params_checkbox,
            outputs=params_accordion
        )

        with gr.Row():
            prompt_input = gr.Textbox(
                label="Enter Your Prompt",
                lines=10,
                scale=4,
                value="Explain the importance of low-rank adaptation (LoRA) for fine-tuning LLMs in simple terms."
            )
        with gr.Row():
            generate_btn = gr.Button("Generate Responses", variant="primary", scale=3)
            stop_btn = gr.Button("Stop", variant="stop", scale=1)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Response A")
                response_a_output = gr.Textbox(label="Response A", interactive=False, lines=25)
                btn_a_better = gr.Button("A is better 👍")
            with gr.Column():
                gr.Markdown("### Response B")
                # NEW: Add a switch to enable manual input for Response B
                manual_b_switch = gr.Checkbox(
                    label="Input Response B Manually",
                    value=False,
                    info="Enable to type or paste your own response for comparison."
                )
                response_b_output = gr.Textbox(label="Response B", interactive=False, lines=25)
                btn_b_better = gr.Button("B is better 👍")

        # NEW: Add event handler to make the Response B textbox editable based on the switch
        manual_b_switch.change(
            fn=lambda is_manual: gr.update(interactive=is_manual),
            inputs=manual_b_switch,
            outputs=response_b_output
        )

        with gr.Row():
            comment_input = gr.Textbox(
                label="Comment / Rationale",
                placeholder="Optional: Explain why you made this choice...",
                lines=2,
                interactive=True
            )

        with gr.Row():
            btn_both_bad = gr.Button("Both are bad 👎")

        # MODIFIED: Added `manual_b_switch` to the inputs list for the generate button.
        gen_event = generate_btn.click(
            fn=generate_stream,
            inputs=[api_url_input, prompt_input, override_params_checkbox, max_tokens_slider, temp_a_slider,
                    top_p_a_slider, temp_b_slider, top_p_b_slider, manual_b_switch],
            outputs=[response_a_output, response_b_output, status_display, stored_prompt, stored_response_a,
                     stored_response_b]
        )

        stop_btn.click(
            fn=stop_inference,
            inputs=None,
            outputs=status_display,
            cancels=[gen_event]
        )

        # MODIFIED: Save buttons now read directly from the response textboxes to capture manual input.
        # `stored_prompt` is used to ensure the correct prompt is saved even if the user edits the input box later.
        # `stored_response_a` is used as it's the definitive generated response.
        # `response_b_output` is used to capture either the generated or manually entered text.
        btn_a_better.click(
            fn=lambda p, a, b, c: save_preference(p, a, b, "A is better", c),
            inputs=[stored_prompt, stored_response_a, response_b_output, comment_input],
            outputs=[status_display, response_a_output, response_b_output, comment_input]
        )
        btn_b_better.click(
            fn=lambda p, a, b, c: save_preference(p, a, b, "B is better", c),
            inputs=[stored_prompt, stored_response_a, response_b_output, comment_input],
            outputs=[status_display, response_a_output, response_b_output, comment_input]
        )
        btn_both_bad.click(
            fn=lambda p, a, b, c: save_preference(p, a, b, "Both are bad", c),
            inputs=[stored_prompt, stored_response_a, response_b_output, comment_input],
            outputs=[status_display, response_a_output, response_b_output, comment_input]
        )

    return demo


if __name__ == "__main__":
    try:
        import requests
        import gradio
    except ImportError:
        print("Required libraries not found. Please install them:")
        print("pip install requests gradio")
        exit()

    ui = build_ui()
    ui.launch()
