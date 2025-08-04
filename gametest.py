import gradio as gr
import json
import os
import math

# Define the number of parts to split the file into
NUM_PARTS = 6


def split_json_file(uploaded_file):
    """
    This function takes an uploaded file object from Gradio, reads it as JSON,
    and splits the top-level list into 6 smaller JSON files.
    """
    if uploaded_file is None:
        return "Error: Please upload a JSON file first.", None

    input_filepath = uploaded_file.name

    try:
        with open(input_filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return "Error: The uploaded file is not a valid JSON file. Please check its format.", None
    except Exception as e:
        return f"An unexpected error occurred while reading the file: {e}", None

    if not isinstance(data, list):
        return "Error: The JSON file's top-level structure must be a list (an array of objects like `[...]`).", None

    total_items = len(data)
    if total_items < NUM_PARTS:
        return f"Error: The JSON file contains only {total_items} items. It cannot be split into {NUM_PARTS} parts.", None

    items_per_part = math.ceil(total_items / NUM_PARTS)

    output_dir = "split_output"
    os.makedirs(output_dir, exist_ok=True)

    output_paths = []

    for i in range(NUM_PARTS):
        start_index = i * items_per_part
        end_index = start_index + items_per_part

        part_data = data[start_index:end_index]

        if not part_data:
            continue

        output_filename = os.path.join(output_dir, f"part_{i + 1}.json")
        output_paths.append(output_filename)

        try:
            with open(output_filename, 'w', encoding='utf-8') as f_out:
                json.dump(part_data, f_out, indent=4)
        except Exception as e:
            return f"Error writing to file {output_filename}: {e}", None

    success_message = (
        f"✅ Success! The file was split into {len(output_paths)} parts with approximately {items_per_part} items each.\n"
        f"Files saved in the '{output_dir}' directory."
    )

    return success_message, output_paths


# --- Build the Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # ✂️ Huge JSON File Splitter
        Upload a large JSON file that contains a top-level list (array). 
        This tool will split it into 6 smaller JSON files.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="Upload JSON File",
                file_types=['.json']
            )
            split_button = gr.Button("Split into 6 Parts", variant="primary")

        with gr.Column(scale=2):
            status_output = gr.Textbox(
                label="Status",
                # The 'info' argument was removed from the line below
                interactive=False
            )
            files_output = gr.Files(
                label="Download Split Files"
                # The 'info' argument was removed from the line below
            )

    split_button.click(
        fn=split_json_file,
        inputs=file_input,
        outputs=[status_output, files_output]
    )

    # --- This Example part might also cause issues on very old Gradio versions.
    # --- If you get another error, try commenting out this entire gr.Examples block.
    gr.Examples(
        examples=[
            [{"fn": lambda: create_dummy_json(100), "file_count": "single"}]
        ],
        inputs=file_input,
        label="Example (Click to use a dummy 100-item JSON file)"
    )


def create_dummy_json(num_items):
    """Helper function for the Gradio example to create a dummy JSON file."""
    dummy_data = [{"id": i, "data": f"item_{i}"} for i in range(num_items)]
    filepath = "dummy_example.json"
    with open(filepath, "w") as f:
        json.dump(dummy_data, f, indent=4)
    return filepath


if __name__ == "__main__":
    demo.launch()