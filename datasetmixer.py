import gradio as gr
import json
import random
import tempfile
import os


# --- Core Logic ---

def shuffle_file(uploaded_file):
    """
    Reads a dataset file (e.g., ShareGPT, DPO), shuffles the entries,
    and returns a path to a new temporary file. It robustly handles both
    standard JSON (list of objects) and JSONL (one object per line).
    """
    if uploaded_file is None:
        return "Please upload a file first.", None

    filepath = uploaded_file.name
    _, file_extension = os.path.splitext(filepath)
    file_extension = file_extension.lower()

    if file_extension not in ['.json', '.jsonl']:
        return f"❌ Error: Unsupported file type '{file_extension}'. Please upload a .json or .jsonl file.", None

    try:
        data = []
        with open(filepath, 'r', encoding='utf-8') as f:
            # For JSONL, the logic is straightforward: one JSON object per line.
            if file_extension == '.jsonl':
                for i, line in enumerate(f, 1):
                    if line.strip(): # Skip empty lines
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid JSON on line {i} in the .jsonl file.")

            # For JSON, we make it more robust.
            elif file_extension == '.json':
                try:
                    # First, try to load as a standard JSON file (could be a list or a single object).
                    loaded_json = json.load(f)
                    if isinstance(loaded_json, list):
                        # This is the expected format for ShareGPT: a list of conversations.
                        data = loaded_json
                    else:
                        # This handles a file with a single JSON object (like a single DPO entry).
                        data = [loaded_json]
                except json.JSONDecodeError:
                    # If the above fails, it's likely a JSONL-formatted file saved with a .json extension.
                    # We rewind the file and read it line by line as a fallback.
                    f.seek(0)
                    for i, line in enumerate(f, 1):
                        if line.strip():
                            try:
                                data.append(json.loads(line))
                            except json.JSONDecodeError:
                                raise ValueError(f"File has .json extension but seems to be JSONL. Found invalid JSON on line {i}.")

        if not data:
            return "⚠️ Warning: The file is empty or contains no valid data. Nothing to shuffle.", None

        original_count = len(data)

        # Shuffle the list of records in-place. This works regardless of the record's internal structure.
        random.shuffle(data)

        # --- Create a temporary file with the correct extension to store the shuffled output ---
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=file_extension, encoding="utf-8") as temp_f:
            if file_extension == '.json':
                # If the original was a single object, we save it as a list of one.
                # Otherwise, we preserve the list format.
                json.dump(data, temp_f, indent=2, ensure_ascii=False)
            elif file_extension == '.jsonl':
                for item in data:
                    # Write each JSON object as a new line
                    temp_f.write(json.dumps(item, ensure_ascii=False) + '\n')

            output_filepath = temp_f.name

        status_message = f"✅ Success! Shuffled {original_count} entries from your {file_extension.upper()} file."
        return status_message, output_filepath

    except (json.JSONDecodeError, TypeError, ValueError) as e:
        # Catch specific, expected errors for clearer feedback
        return f"❌ Error: {e}", None
    except Exception as e:
        # Catch any other unexpected errors
        return f"❌ An unexpected error occurred: {e}", None


# --- Create dummy example files for the Gradio interface ---

def create_example_files():
    """Creates sample ShareGPT and DPO style files for demonstration."""
    # 1. ShareGPT Example Data
    sharegpt_data = [
        {"id": "convo_1", "conversations": [{"from": "human", "value": "Hello"}, {"from": "gpt", "value": "Hi there!"}]},
        {"id": "convo_2", "conversations": [{"from": "human", "value": "Capital of France?"}, {"from": "gpt", "value": "Paris."}]},
        {"id": "convo_3", "conversations": [{"from": "human", "value": "Poem about a cat."}, {"from": "gpt", "value": "Sunbeam nap, a gentle purr."}]}
    ]
    # Create JSONL ShareGPT example
    sharegpt_jsonl_path = "example_sharegpt.jsonl"
    with open(sharegpt_jsonl_path, "w", encoding="utf-8") as f:
        for item in sharegpt_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    # 2. DPO Example Data
    dpo_data = [
        {"prompt": "What is DPO?", "chosen": "DPO is Direct Preference Optimization.", "rejected": "DPO is a type of donut."},
        {"prompt": "Who wrote 'Hamlet'?", "chosen": "William Shakespeare wrote 'Hamlet'.", "rejected": "Charles Dickens wrote 'Hamlet'."},
        {"prompt": "Explain gravity.", "chosen": "Gravity is the force by which a planet or other body draws objects toward its center.", "rejected": "Gravity is what makes balloons float up."}
    ]
    # Create JSONL DPO example
    dpo_jsonl_path = "example_dpo.jsonl"
    with open(dpo_jsonl_path, "w", encoding="utf-8") as f:
        for item in dpo_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return [sharegpt_jsonl_path, dpo_jsonl_path]


example_paths = create_example_files()

# --- Gradio UI ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔀 Dataset Shuffler (JSON & JSONL)
        Upload your dataset in **JSON** or **JSONL** format. The tool will randomly shuffle the order of the entries and provide a new, shuffled file for you to download.

        This works for various formats, including **ShareGPT** conversations and **DPO**-style `{"prompt": ..., "chosen": ..., "rejected": ...}` entries.
        - **JSON format:** Supports a standard list `[...]` of objects. If it fails, it will try reading as line-delimited JSON.
        - **JSONL format:** Each line must be a separate, valid JSON object `{...}`.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.File(
                label="Upload JSON or JSONL File",
                file_types=[".json", ".jsonl"]
            )
            shuffle_button = gr.Button("Shuffle Dataset", variant="primary")

            gr.Examples(
                examples=example_paths,
                inputs=input_file,
                label="Click an example to start"
            )

        with gr.Column(scale=1):
            status_output = gr.Textbox(
                label="Status",
                interactive=False,
                placeholder="Upload a file and click shuffle..."
            )
            output_file = gr.File(
                label="Download Shuffled File",
                interactive=False
            )

    # Connect the button to the function
    shuffle_button.click(
        fn=shuffle_file,
        inputs=input_file,
        outputs=[status_output, output_file]
    )

if __name__ == "__main__":
    demo.launch()