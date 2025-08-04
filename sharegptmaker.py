import gradio as gr
import json
import os
import tempfile

# --- Persistent Storage ---
# The file where the dataset will be saved automatically
DATASET_FILE = "sharegpt_persistent_dataset.json"


def load_dataset():
    """
    Loads the dataset from the JSON file if it exists.
    Returns an empty list if the file doesn't exist or is invalid.
    """
    if os.path.exists(DATASET_FILE):
        try:
            with open(DATASET_FILE, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
                # Basic validation to ensure it's a list
                if isinstance(dataset, list):
                    return dataset
                else:
                    print(f"Warning: Data in {DATASET_FILE} is not a list. Starting fresh.")
                    return []
        except (json.JSONDecodeError, IOError) as e:
            print(f"Error loading {DATASET_FILE}: {e}. Starting with an empty dataset.")
            return []
    return []


def save_dataset(dataset):
    """
    Saves the entire dataset to the JSON file.
    """
    with open(DATASET_FILE, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


# --- App Logic Functions ---

def add_to_dataset(human_input, gpt_input, current_dataset):
    """
    Adds a new human-gpt pair to the dataset and saves it to the persistent file.
    """
    if not human_input.strip() or not gpt_input.strip():
        gr.Warning("Both Human and GPT inputs are required.")
        status_message = f"⚠️ **Action Failed!** Both fields are required. Dataset still has {len(current_dataset)} entries."
        return current_dataset, status_message, human_input, gpt_input

    new_entry = {
        "conversations": [
            {"from": "human", "value": human_input},
            {"from": "gpt", "value": gpt_input}
        ]
    }

    current_dataset.append(new_entry)

    # --- This is the key change for persistence ---
    save_dataset(current_dataset)
    # ---------------------------------------------

    status_message = f"✅ **Success!** Entry added and saved. Dataset now has **{len(current_dataset)}** entries."

    return current_dataset, status_message, "", ""


def export_dataset(dataset):
    """
    Exports a copy of the current dataset state for download.
    """
    if not dataset:
        gr.Info("Dataset is empty. Add some entries before exporting.")
        return None

    # Use a temporary file to avoid interfering with the main dataset file
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
        filepath = f.name

    gr.Info(f"Dataset exported successfully! You can now download the file.")

    final_filepath = filepath.replace(os.path.basename(filepath), "sharegpt_dataset_export.json")
    os.rename(filepath, final_filepath)

    return final_filepath


# --- Gradio App Interface ---

with gr.Blocks(theme=gr.themes.Soft(), title="Persistent ShareGPT Dataset Creator") as demo:
    # --- Load initial data and set up state ---
    initial_dataset = load_dataset()
    dataset_state = gr.State(initial_dataset)

    gr.Markdown(
        """
        # 📝 Persistent ShareGPT Dataset Creator
        Enter a human prompt and a GPT response. Your entries are **saved automatically** between sessions.
        """
    )

    with gr.Row():
        human_input = gr.Textbox(lines=8, label="Human Input",
                                 placeholder="Enter the human's prompt or question here...")
        gpt_input = gr.Textbox(lines=8, label="GPT Output", placeholder="Enter the model's response here...")

    add_btn = gr.Button("Add to Dataset", variant="primary")

    gr.Markdown("---")

    # Display the initial status based on the loaded data
    initial_status = f"Dataset loaded with **{len(initial_dataset)}** entries. Your work is saved to `{DATASET_FILE}`."
    status_display = gr.Markdown(initial_status)

    gr.Markdown("---")

    with gr.Row():
        export_btn = gr.Button("Export a Copy", variant="secondary")
        download_link = gr.File(label="Download Exported File", interactive=False)

    # --- Event Handlers ---

    add_btn.click(
        fn=add_to_dataset,
        inputs=[human_input, gpt_input, dataset_state],
        outputs=[dataset_state, status_display, human_input, gpt_input]
    )

    export_btn.click(
        fn=export_dataset,
        inputs=[dataset_state],
        outputs=[download_link]
    )

if __name__ == "__main__":
    demo.launch()