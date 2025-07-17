import gradio as gr
import json
import random
import tempfile
import os


# --- Core Logic ---

def shuffle_file(uploaded_file):
    """
    Reads a ShareGPT-style JSON or JSONL file, shuffles the list of
    conversations/entries, and returns a path to the new temporary file.
    """
    if uploaded_file is None:
        return "Please upload a file first.", None

    filepath = uploaded_file.name
    # Determine file type from its extension
    _, file_extension = os.path.splitext(filepath)
    file_extension = file_extension.lower()

    if file_extension not in ['.json', '.jsonl']:
        return f"❌ Error: Unsupported file type '{file_extension}'. Please upload a .json or .jsonl file.", None

    try:
        data = []
        # Read the file based on its extension
        if file_extension == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # The core assumption for JSON is a single list of conversations
            if not isinstance(data, list):
                raise TypeError("The JSON file is not in the expected format. It should be a list [...] of objects.")

        elif file_extension == '.jsonl':
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    if line.strip(): # Skip empty lines
                        try:
                            data.append(json.loads(line))
                        except json.JSONDecodeError:
                            # Raise a more informative error for bad JSONL
                            raise ValueError(f"Invalid JSON on line {i} in the .jsonl file.")

        if not data:
            return "⚠️ Warning: The file is empty or contains no valid data. Nothing to shuffle.", None

        original_count = len(data)

        # Shuffle the list in-place
        random.shuffle(data)

        # --- Create a temporary file with the correct extension to store the shuffled output ---
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=file_extension, encoding="utf-8") as temp_f:
            if file_extension == '.json':
                # Use ensure_ascii=False to correctly handle non-English characters
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
    """Creates sample ShareGPT-style JSON and JSONL files for demonstration."""
    example_data = [
        {
            "id": "convo_1",
            "conversations": [
                {"from": "human", "value": "Hello, who are you?"},
                {"from": "gpt", "value": "I am a helpful assistant."}
            ]
        },
        {
            "id": "convo_2",
            "conversations": [
                {"from": "human", "value": "What is the capital of France?"},
                {"from": "gpt", "value": "The capital of France is Paris."}
            ]
        },
        {
            "id": "convo_3",
            "conversations": [
                {"from": "human", "value": "Write a short poem about a cat."},
                {"from": "gpt",
                 "value": "Sunbeam nap on a cozy mat,\nA twitching ear, a gentle pat,\nDreaming deep, the happy cat."}
            ]
        }
    ]
    # Create JSON example
    json_path = "example_sharegpt.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(example_data, f, indent=2)

    # Create JSONL example
    jsonl_path = "example_sharegpt.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in example_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    return [json_path, jsonl_path]


example_paths = create_example_files()

# --- Gradio UI ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔀 Dataset Shuffler (JSON & JSONL)
        Upload your dataset in **JSON** or **JSONL** format. The tool will randomly shuffle the order of the entries and provide a new, shuffled file for you to download.

        - **JSON format:** The root of the file must be a single list `[...]` of objects.
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
                interactive=False
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