import gradio as gr
import json
import os
from collections import OrderedDict

# MODIFIED FUNCTION to handle a list of files
def convert_to_sharegpt(temp_files):
    """
    Converts one or more JSON files from the given format to the ShareGPT format,
    aggregating all conversations into a single list.

    Args:
        temp_files: A list of temporary file objects from Gradio's file upload.

    Returns:
        A list of all conversations in ShareGPT format, and a filepath to the
        single converted file for download.
    """
    # Handle the case where no files are uploaded
    if not temp_files:
        return None, None

    # This list will aggregate data from all valid files
    all_sharegpt_data = []

    # Iterate over the list of uploaded files
    for temp_file in temp_files:
        try:
            with open(temp_file.name, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            # Use gr.Warning to show a non-blocking message to the user and continue
            gr.Warning(f"Skipping file '{os.path.basename(temp_file.name)}' due to error: {e}")
            continue # Move to the next file

        # The core processing logic for a single file's data remains the same
        for conversation_data in data:
            messages = conversation_data.get("chat", {}).get("history", {}).get("messages", {})
            if not messages:
                continue

            root = None
            for msg_id, msg in messages.items():
                if msg.get("parentId") is None:
                    root = msg
                    break

            if root is None:
                continue

            conversation_history = []
            current_msg = root
            while current_msg:
                role = current_msg.get("role")
                content = ""
                if isinstance(current_msg.get("content_list"), list):
                    content = "".join(item.get("content", "") for item in current_msg["content_list"])
                elif current_msg.get("content"):
                    content = current_msg.get("content")

                if role and content:
                    conversation_history.append({"role": role, "content": content})

                children_ids = current_msg.get("childrenIds", [])
                if not children_ids:
                    break
                next_child_id = children_ids[0]
                current_msg = messages.get(next_child_id)

            sharegpt_conversations = []
            for message in conversation_history:
                role = message.get("role")
                content = message.get("content", "")
                if role == "user":
                    sharegpt_conversations.append({"from": "human", "value": content})
                elif role == "assistant":
                    sharegpt_conversations.append({"from": "gpt", "value": content})

            if sharegpt_conversations:
                # Add the processed conversation to our main list
                all_sharegpt_data.append({
                    "id": conversation_data.get("id"),
                    "conversations": sharegpt_conversations
                })

    # If no data was successfully converted (e.g., all files had errors or were empty)
    if not all_sharegpt_data:
        gr.Warning("No valid conversations could be extracted from the uploaded files.")
        return None, None

    # Save the aggregated data to a new file for download
    output_filename = "converted_sharegpt_multiple.json"
    with open(output_filename, 'w', encoding='utf-8') as f:
        json.dump(all_sharegpt_data, f, indent=2)

    return all_sharegpt_data, output_filename


with gr.Blocks() as demo:
    gr.Markdown(
        """
        # JSON to ShareGPT Converter (Multi-File)
        Upload one or more JSON files in the specified format. The tool will process all of them
        and combine the results into a single ShareGPT-formatted file for download.
        """
    )

    with gr.Row():
        with gr.Column():
            # MODIFIED: Added file_count="multiple" to allow multiple uploads
            file_input = gr.File(label="Upload JSON Files", file_count="multiple")
            convert_button = gr.Button("Convert")

        with gr.Column():
            json_output = gr.JSON(label="ShareGPT Output")
            download_button = gr.DownloadButton(label="Download Converted File", interactive=False)


    # MODIFIED: The function now correctly handles a list of files
    def process_and_update(list_of_files):
        if not list_of_files:
            # Return None for the JSON output, and an update to make the button non-interactive
            gr.Info("Please upload at least one file.")
            return None, gr.update(interactive=False, value=None)

        # The core conversion function now handles the list directly
        sharegpt_json, output_path = convert_to_sharegpt(list_of_files)

        # If the conversion resulted in no data, handle UI updates
        if not sharegpt_json:
             return None, gr.update(interactive=False, value=None)

        # Return the JSON data and an update for the button with the new file path, making it interactive
        return sharegpt_json, gr.update(value=output_path, interactive=True)


    convert_button.click(
        fn=process_and_update,
        inputs=file_input,
        outputs=[json_output, download_button]
    )

if __name__ == "__main__":
    demo.launch()