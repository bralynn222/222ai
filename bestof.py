import gradio as gr
import json
import os
import time
from openai import OpenAI, APIError, AuthenticationError

# --- Configuration ---
OUTPUT_FILENAME = "best_of_dataset.json"

# --- Judge Prompt Template (Unchanged) ---
JUDGE_PROMPT_TEMPLATE = (
    'You are an impartial AI assistant. Your task is to evaluate two responses (A and B) based on a user prompt. '
    'Your evaluation should be based on adherence to the user\'s request (75% weight) and the accuracy and depth of the information (25% weight). '
    'You may also consider factors like clarity, conciseness, and formatting.\n\n'
    'First, provide a brief but clear rationale for your decision, explaining your reasoning for the choice. '
    'Then, on a new line, you MUST state the winner using ONE of the following formats:\n'
    '   - "Key-WINBOY=A" if Response A is better.\n'
    '   - "Key-WINBOY=B" if Response B is better.\n'
    '   - "Key-WINBOY=BAD" if BOTH responses are of poor quality, fail to follow the prompt\'s instructions, or are otherwise incorrect. Do NOT choose a winner in this case.\n\n'
    'Do not deviate from these key formats. Your entire response must contain your reasoning followed by the key.\n\n'
    '--- USER PROMPT ---\n{prompt}\n\n--- RESPONSE A ---\n{response_a}\n\n'
    '--- RESPONSE B ---\n{response_b}\n\n--- YOUR EVALUATION AND CHOICE ---'
)


# --- update_models_from_api function (Unchanged) ---
def update_models_from_api(api_url, api_key):
    # This function is correct and remains the same.
    if not api_url or not api_key: return gr.Dropdown(choices=[], value=None, interactive=False,
                                                      label="API URL and Key required")
    try:
        client = OpenAI(api_key=api_key, base_url=api_url)
        models_response = client.models.list()
        all_models = sorted([model.id for model in models_response.data])
        if not all_models:
            gr.Warning(f"Connected to {api_url}, but no models were found.")
            return gr.Dropdown(choices=[], value=None, interactive=True, label="No models found at endpoint")
        preferred_defaults = ["gpt-4o", "gpt-4-turbo", "gpt-4-turbo-preview", "gpt-4", "claude-3-opus-20240229",
                              "gpt-3.5-turbo"]
        default_model = all_models[0]
        for model in preferred_defaults:
            if model in all_models: default_model = model; break
        gr.Info(f"Success! Found {len(all_models)} models at {api_url}.")
        return gr.Dropdown(choices=all_models, value=default_model, interactive=True, label="Select Judge Model")
    except AuthenticationError:
        gr.Warning("Authentication Error: Invalid API Key or URL.");
        return gr.Dropdown(choices=[], value=None, interactive=False, label="Authentication Failed")
    except Exception as e:
        gr.Warning(f"An error occurred: {e}");
        return gr.Dropdown(choices=[], value=None, interactive=False, label="Error fetching models")


# --- THIS IS THE FINAL, ROBUST VERSION OF THE FUNCTION ---
def get_judge_verdict(client, judge_model, prompt, response_a, response_b):
    """
    Sends a pair of responses to the chosen judge model and gets the evaluation.
    This function uses an "intelligent fallback" to handle API parameter differences.
    """
    judge_prompt = JUDGE_PROMPT_TEMPLATE.format(prompt=prompt, response_a=response_a, response_b=response_b)

    try:
        # Attempt 1: The "Ideal" call with deterministic parameters.
        print(f"Attempting ideal API call for model '{judge_model}'...")
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            temperature=0,
            max_tokens=1024
        )
        return response.choices[0].message.content

    except APIError as e:
        # If the ideal call fails with a parameter error (400), fall back to "safe mode".
        if e.status_code == 400:
            print(
                f"Warning: Ideal API call failed with a parameter error (400). Retrying in 'safe mode'. Details: {e.body}")
            try:
                # Attempt 2: The "Safe Mode" call with only essential parameters.
                response = client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": judge_prompt}]
                )
                return response.choices[0].message.content
            except Exception as retry_e:
                return f"Error: The 'safe mode' retry also failed. Details: {retry_e}"
        else:
            # If it's another type of error (e.g., server error 500), return it.
            return f"Error: A non-parameter API error occurred. Details: {e}"
    except Exception as e:
        return f"Error: An unexpected error occurred. Details: {e}"


# --- process_datasets function (Unchanged) ---
def process_datasets(api_url, api_key, judge_model, file_a, file_b, progress=gr.Progress(track_tqdm=True)):
    # This function is correct and remains the same.
    if not api_url: raise gr.Error("API URL is required.")
    if not api_key: raise gr.Error("API Key is required.")
    if not judge_model: raise gr.Error("Please select a Judge Model from the dropdown.")
    if not file_a or not file_b: raise gr.Error("Please upload both Dataset A and Dataset B.")
    try:
        client = OpenAI(api_key=api_key, base_url=api_url)
    except Exception as e:
        raise gr.Error(f"Failed to initialize API client. Check your URL and Key. Error: {e}")
    try:
        with open(file_a.name, 'r', encoding='utf-8') as f:
            data_a = json.load(f)
        with open(file_b.name, 'r', encoding='utf-8') as f:
            data_b = json.load(f)
    except json.JSONDecodeError:
        raise gr.Error("One of the files is not a valid JSON.")
    except Exception as e:
        raise gr.Error(f"Error reading files: {e}")
    num_pairs = min(len(data_a), len(data_b))
    if num_pairs == 0: raise gr.Error("One or both datasets are empty.")
    best_of_conversations, wins_a, wins_b, bads, errors = [], 0, 0, 0, 0
    for i in progress.tqdm(range(num_pairs), desc="Comparing Pairs"):
        conv_a, conv_b = data_a[i], data_b[i]
        try:
            prompt = conv_a['conversations'][0]['value']
            response_a = conv_a['conversations'][1]['value']
            response_b = conv_b['conversations'][1]['value']
            if prompt != conv_b['conversations'][0]['value']:
                errors += 1;
                print(f"Warning: Mismatched prompts at index {i}. Skipping.");
                continue
        except (KeyError, IndexError):
            errors += 1;
            print(f"Warning: Malformed conversation at index {i}. Skipping.");
            continue
        verdict = get_judge_verdict(client, judge_model, prompt, response_a, response_b)
        if "Key-WINBOY=A" in verdict:
            wins_a += 1; best_of_conversations.append(conv_a)
        elif "Key-WINBOY=B" in verdict:
            wins_b += 1; best_of_conversations.append(conv_b)
        elif "Key-WINBOY=BAD" in verdict:
            bads += 1
        else:
            errors += 1; print(f"Warning: Could not parse verdict for pair {i + 1}. Verdict:\n{verdict}")
    with open(OUTPUT_FILENAME, 'w', encoding='utf-8') as f:
        json.dump(best_of_conversations, f, indent=2, ensure_ascii=False)
    final_summary = (
        f"✅ Processing Complete!\n\n"
        f"Total Pairs Processed: {num_pairs}\n"
        f"Judge Model Used: '{judge_model}'\n"
        f"New Dataset Size: {len(best_of_conversations)} conversations\n\n"
        f"--- FINAL RESULTS ---\n"
        f"Dataset A Wins: {wins_a} ({wins_a / num_pairs:.2%})\n"
        f"Dataset B Wins: {wins_b} ({wins_b / num_pairs:.2%})\n"
        f"Both Marked as Bad: {bads} ({bads / num_pairs:.2%})\n"
        f"Errors/Skipped: {errors} ({errors / num_pairs:.2%})\n\n"
        f"The winning dataset has been saved as '{OUTPUT_FILENAME}' and is available for download below."
    )
    return final_summary, gr.File(value=OUTPUT_FILENAME)


# --- Gradio UI Definition (Unchanged) ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # The UI code is correct and remains the same.
    gr.Markdown("# ShareGPT \"Best-Of\" Dataset Creator")
    gr.Markdown(
        """
        1.  Enter the **API Base URL** for your service (e.g., OpenAI, OpenRouter).
        2.  Provide the corresponding **API Key**. The app will fetch all available models.
        3.  Select the **Judge Model** from the dropdown.
        4.  Upload your two ShareGPT-formatted JSON files.
        5.  Click **Create Best-Of Dataset** and watch the progress bar.
        """
    )
    with gr.Row():
        with gr.Column(scale=1):
            api_url_input = gr.Textbox(label="API Base URL", value="https://api.openai.com/v1",
                                       info="E.g., https://api.openai.com/v1 or https://openrouter.ai/api/v1")
            api_key_input = gr.Textbox(label="API Key", type="password", placeholder="Enter your API key here")
            model_dropdown = gr.Dropdown(label="Select Judge Model", info="Models load after entering URL and Key.",
                                         interactive=False)
            file_input_a = gr.File(label="Upload Dataset A (.json)")
            file_input_b = gr.File(label="Upload Dataset B (.json)")
            process_button = gr.Button("Create Best-Of Dataset", variant="primary")
        with gr.Column(scale=2):
            output_report = gr.Textbox(label="Final Report", lines=15, interactive=False)
            file_output = gr.File(label="Download Best-of Dataset")
    api_url_input.blur(fn=update_models_from_api, inputs=[api_url_input, api_key_input], outputs=[model_dropdown])
    api_key_input.blur(fn=update_models_from_api, inputs=[api_url_input, api_key_input], outputs=[model_dropdown])
    process_button.click(
        fn=process_datasets,
        inputs=[api_url_input, api_key_input, model_dropdown, file_input_a, file_input_b],
        outputs=[output_report, file_output]
    )

if __name__ == "__main__":
    demo.launch()