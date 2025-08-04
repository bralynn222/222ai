import gradio as gr
import json
import tempfile
import os
from transformers import AutoTokenizer

# --- Configuration ---
# CHANGE 1: Use a public, non-gated Llama 3 fine-tune to load the tokenizer.
# This avoids the "gated repo" error. The tokenizer is the same as the base model.
MODEL_ID = "bralynn/ed2"

print(f"Loading tokenizer from public repo '{MODEL_ID}'...")
try:
    # Load the tokenizer once when the app starts
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
except Exception as e:
    print(f"FATAL: Could not load tokenizer.")
    print(f"Error: {e}")
    tokenizer = None


# --- Core Logic ---

def apply_llama3_template_to_row(row: dict) -> dict:
    """
    Applies the Llama 3 chat template to a single DPO data row.
    A row is a dictionary with 'prompt', 'chosen', and 'rejected' keys.
    """
    if 'prompt' not in row or 'chosen' not in row or 'rejected' not in row:
        raise ValueError("Each JSON object must contain 'prompt', 'chosen', and 'rejected' keys.")

    prompt_messages = [{"role": "user", "content": row['prompt']}]
    chosen_messages = prompt_messages + [{"role": "assistant", "content": row['chosen']}]
    rejected_messages = prompt_messages + [{"role": "assistant", "content": row['rejected']}]

    formatted_prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )
    formatted_chosen = tokenizer.apply_chat_template(chosen_messages, tokenize=False)
    formatted_rejected = tokenizer.apply_chat_template(rejected_messages, tokenize=False)

    return {
        "prompt": formatted_prompt,
        "chosen": formatted_chosen,
        "rejected": formatted_rejected
    }


def process_dpo_file(uploaded_file):
    """
    Main function triggered by the Gradio button.
    Reads an uploaded file, processes it, and returns results.

    CHANGE 2: This version is more robust. It tries to parse the file as standard JSON first.
    If that fails, it assumes the file is JSON Lines and tries to parse it line-by-line.
    This fixes the 'Extra data' JSONDecodeError.
    """
    if not tokenizer:
        raise gr.Error("Tokenizer failed to load. Please check the console for errors.")

    if uploaded_file is None:
        return None, "Please upload a file first."

    filepath = uploaded_file.name
    data = []

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # First, try to load as a standard JSON (a single list of objects)
            try:
                data = json.load(f)
                if not isinstance(data, list):
                    # Handle case where a single JSON object is not in a list
                    data = [data]
            except json.JSONDecodeError:
                # If it fails, it might be a JSON Lines file. Rewind and try that.
                print("Could not parse as standard JSON... trying as JSON Lines.")
                f.seek(0)  # Go back to the start of the file
                data = [json.loads(line) for line in f if line.strip()]

    except Exception as e:
        # If both attempts fail, the file is likely malformed.
        raise gr.Error(f"Error reading or parsing file. Please ensure it's a valid JSON or JSONL file. Details: {e}")

    # Process each row
    processed_data = []
    for row in data:
        try:
            processed_data.append(apply_llama3_template_to_row(row))
        except ValueError as e:
            raise gr.Error(f"Error processing row: {row}. Details: {e}")

    # Save the processed data to a temporary .jsonl file for download
    temp_dir = tempfile.gettempdir()
    output_path = os.path.join(temp_dir, "formatted_dpo_data.jsonl")

    with open(output_path, 'w', encoding='utf-8') as f:
        for item in processed_data:
            f.write(json.dumps(item) + '\n')

    # Return the path to the downloadable file and a preview for the UI
    preview = processed_data[:5]

    return output_path, preview


# --- Gradio UI ---

with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
    gr.Markdown(
        """
        # 🦙 Llama 3 DPO Dataset Formatter
        Upload your DPO dataset to apply the official Llama 3 chat template.

        **Input Format:**
        Your file should be a valid JSON or JSONL file. Each object must contain three keys:
        - `prompt`: The text of the user's message.
        - `chosen`: The text of the preferred assistant response.
        - `rejected`: The text of the dispreferred assistant response.

        **Output Format:**
        The tool will generate a `.jsonl` file where all special tokens (`<|begin_of_text|>`, `<|start_header_id|>`, `<|eot_id|>`, etc.) have been correctly added to each field. This file is ready for use with DPO training libraries like TRL's `DPOTrainer`.
        """
    )

    with gr.Row():
        input_file = gr.File(
            label="Upload DPO JSON/JSONL File",
            file_types=['.json', '.jsonl']
        )

    process_button = gr.Button("Apply Llama 3 Chat Template", variant="primary")

    gr.Markdown("---")
    gr.Markdown("## Output")

    with gr.Row():
        output_file = gr.File(
            label="Download Formatted File",
            interactive=False
        )
        output_preview = gr.JSON(
            label="Preview of First 5 Processed Rows"
        )

    process_button.click(
        fn=process_dpo_file,
        inputs=[input_file],
        outputs=[output_file, output_preview]
    )

if __name__ == "__main__":
    if tokenizer is None:
        print("\nApplication cannot start because the tokenizer could not be loaded.")
    else:
        print("\nTokenizer loaded successfully. Starting Gradio app...")
        demo.launch()