import gradio as gr
import json
import tempfile
import os


# --- 1. Core Logic Function ---
# This function takes the uploaded files, processes them, and returns the results.

def combine_sharegpt_datasets(files):
    """
    Combines multiple ShareGPT-formatted JSON files into one.

    Args:
        files (list): A list of file-like objects from the Gradio File component.

    Returns:
        tuple: A tuple containing:
            - str: A markdown string summarizing the process.
            - str: The file path to the combined JSON for download, or None if failed.
    """
    if not files:
        return "Please upload at least one JSON file.", None

    combined_data = []
    error_log = []
    processed_files_count = 0
    total_conversations_found = 0

    for file_obj in files:
        filename = os.path.basename(file_obj.name)
        try:
            # Gradio file objects have a .name attribute which is the path to a temp file
            with open(file_obj.name, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Validate that the data is a list (standard ShareGPT format)
            if not isinstance(data, list):
                error_log.append(f"❌ **{filename}**: Skipped. File content is not a list as expected.")
                continue

            # Add the conversations from this file to our master list
            combined_data.extend(data)

            # Update stats
            processed_files_count += 1
            conversations_in_file = len(data)
            total_conversations_found += conversations_in_file
            print(f"Successfully processed {filename} with {conversations_in_file} conversations.")

        except json.JSONDecodeError:
            error_log.append(f"❌ **{filename}**: Skipped. Invalid JSON format.")
        except Exception as e:
            error_log.append(f"❌ **{filename}**: Skipped. An unexpected error occurred: {e}")

    # --- Prepare the output ---

    # If no data was successfully processed, return an error message.
    if not combined_data:
        summary = "### ⚠️ Processing Failed\n\nNo valid conversations could be extracted."
        if error_log:
            summary += "\n\n**Errors:**\n" + "\n".join(error_log)
        return summary, None

    # Create a temporary file to save the combined data
    try:
        # We use delete=False because Gradio needs the file to exist after this function returns
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".json", encoding='utf-8') as temp_f:
            # Use indent=2 for human-readability
            json.dump(combined_data, temp_f, indent=2, ensure_ascii=False)
            output_filepath = temp_f.name

        # Create a success summary
        summary = f"### ✅ Processing Complete\n\n"
        summary += f"- **Files Processed Successfully:** {processed_files_count}\n"
        summary += f"- **Total Conversations Combined:** {total_conversations_found}\n"

        if error_log:
            summary += "\n\n**Issues Encountered:**\n" + "\n".join(error_log)

        return summary, output_filepath

    except Exception as e:
        summary = f"### ❌ Error Creating Output File\n\nAn unexpected error occurred: {e}"
        return summary, None


# --- 2. Create Dummy Files for the Example ---
# This helps users understand what to upload without having to find their own files first.

def create_example_files():
    """Creates dummy JSON files for the Gradio example."""
    os.makedirs("examples", exist_ok=True)

    data1 = [
        {"conversations": [{"from": "human", "value": "Hello"}, {"from": "gpt", "value": "Hi there!"}]},
        {"conversations": [{"from": "human", "value": "How are you?"},
                           {"from": "gpt", "value": "I'm a model, I'm fine."}]}
    ]
    with open("examples/dataset1.json", "w") as f:
        json.dump(data1, f, indent=2)

    data2 = [
        {"conversations": [{"from": "human", "value": "What is Gradio?"},
                           {"from": "gpt", "value": "It's a Python library..."}]}
    ]
    with open("examples/dataset2.json", "w") as f:
        json.dump(data2, f, indent=2)

    # Create a file with an invalid format
    with open("examples/invalid_format.json", "w") as f:
        f.write('{"conversations": "this is not a list"}')

    return [
        ["examples/dataset1.json", "examples/dataset2.json"],
        ["examples/dataset1.json", "examples/invalid_format.json"]
    ]


# --- 3. Build the Gradio App Interface ---

with gr.Blocks(theme=gr.themes.Soft(), title="ShareGPT Combiner") as demo:
    gr.Markdown(
        """
        # 📚 ShareGPT Dataset Combiner
        Upload multiple ShareGPT-formatted JSON files. The app will merge them into a single dataset for you to download.

        **Note:** The root of each JSON file should be a list `[...]` of conversation objects.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload ShareGPT JSON Files",
                file_count="multiple",
                file_types=[".json"]
            )
            combine_btn = gr.Button("Combine Datasets", variant="primary")

            gr.Markdown("---")
            gr.Markdown("### Examples")
            example_files = create_example_files()
            gr.Examples(
                examples=example_files,
                inputs=file_input,
                label="Click an example to load files"
            )

        with gr.Column(scale=2):
            summary_output = gr.Markdown(label="Processing Summary")
            file_output = gr.File(label="Download Combined Dataset", interactive=False)

    # Connect the button to the function
    combine_btn.click(
        fn=combine_sharegpt_datasets,
        inputs=file_input,
        outputs=[summary_output, file_output]
    )

if __name__ == "__main__":
    demo.launch()