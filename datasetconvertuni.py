import gradio as gr
import json
import tempfile
import uuid

# --- 1. Define Formats & Create Example Files ---

# Added "Alpaca-Labeled" to the list of formats
FORMATS = ["Alpaca", "ShareGPT", "ChatML", "OpenAI", "DPO", "Alpaca-Labeled"]

# --- Example Data for all formats ---
EXAMPLE_ALPACA_DATA = [
    {"instruction": "What is the capital of Spain?", "input": "", "output": "The capital of Spain is Madrid."},
    {"instruction": "Write a short poem about the moon.", "input": "",
     "output": "Silver orb in velvet night,\nCasting shadows, soft and light,\nSilent watcher, ever bright,\nGuiding sailors with your light."}
]
EXAMPLE_SHAREGPT_DATA = [
    {"id": "conv1", "conversations": [{"from": "human", "value": "What is the capital of France?"},
                                      {"from": "gpt", "value": "The capital of France is Paris."}]},
    {"id": "conv2", "conversations": [{"from": "human", "value": "Give me three ideas for a healthy breakfast."},
                                      {"from": "gpt", "value": "1. Oatmeal\n2. Yogurt\n3. Eggs"},
                                      {"from": "human", "value": "Which is quickest?"},
                                      {"from": "gpt", "value": "Yogurt is the quickest."}]}
]
EXAMPLE_CHATML_DATA = [
    {"messages": [{"role": "system", "content": "You are a helpful AI assistant."},
                  {"role": "user", "content": "What is the capital of France?"},
                  {"role": "assistant", "content": "The capital of France is Paris."}]},
    {"messages": [{"role": "user", "content": "Give me three ideas for a healthy breakfast."},
                  {"role": "assistant", "content": "1. Oatmeal\n2. Yogurt\n3. Eggs"},
                  {"role": "user", "content": "Which is quickest?"},
                  {"role": "assistant", "content": "Yogurt is the quickest."}]}
]
EXAMPLE_OPENAI_DATA = EXAMPLE_CHATML_DATA # Structurally identical for this converter
EXAMPLE_DPO_DATA = [
    {"prompt": "What is the capital of Spain?", "chosen": "The capital of Spain is Madrid.", "rejected": "The capital of Spain is Barcelona."},
    {"prompt": "Write a short poem about the moon.",
     "chosen": "Silver orb in velvet night,\nCasting shadows, soft and light,\nSilent watcher, ever bright,\nGuiding sailors with your light.",
     "rejected": "The moon is a big rock in the sky. It is white. It is not the sun."}
]
# New example data for the "Alpaca-Labeled" format
EXAMPLE_ALPACALABELED_DATA = [
    {
      "instruction": "You will be given a series of words. Output these words in reverse order, with each word on its own line.",
      "input": "Words: ['Hello', 'world'].",
      "output": "['world', 'Hello']",
      "label": "world\nHello"
    },
    {
      "instruction": "Identify the verb in the sentence.",
      "input": "The cat quickly jumped.",
      "output": "'jumped'",
      "label": "jumped"
    }
]

# --- Create example files for all formats ---
example_paths = {
    "Alpaca": "example_alpaca.json",
    "ShareGPT": "example_sharegpt.json",
    "ChatML": "example_chatml.json",
    "OpenAI": "example_openai.json",
    "DPO": "example_dpo.json",
    "Alpaca-Labeled": "example_alpacalabeled.json"
}
with open(example_paths["Alpaca"], "w", encoding="utf-8") as f: json.dump(EXAMPLE_ALPACA_DATA, f, indent=2)
with open(example_paths["ShareGPT"], "w", encoding="utf-8") as f: json.dump(EXAMPLE_SHAREGPT_DATA, f, indent=2)
with open(example_paths["ChatML"], "w", encoding="utf-8") as f: json.dump(EXAMPLE_CHATML_DATA, f, indent=2)
with open(example_paths["OpenAI"], "w", encoding="utf-8") as f: json.dump(EXAMPLE_OPENAI_DATA, f, indent=2)
with open(example_paths["DPO"], "w", encoding="utf-8") as f: json.dump(EXAMPLE_DPO_DATA, f, indent=2)
with open(example_paths["Alpaca-Labeled"], "w", encoding="utf-8") as f: json.dump(EXAMPLE_ALPACALABELED_DATA, f, indent=2)


# --- 2. Conversion Logic Functions ---

def from_alpaca(data):
    # Base prompt combines instruction and input
    def get_prompt(item):
        return item['instruction'] + (f"\n\n{item['input']}" if item.get('input') else "")

    sharegpt_data = [{"id": str(uuid.uuid4()), "conversations": [
        {"from": "human", "value": get_prompt(item)},
        {"from": "gpt", "value": item['output']}]} for item in data]
    chatml_data = [{"messages": [{"role": "system", "content": "You are a helpful assistant."},
                                 {"role": "user", "content": get_prompt(item)},
                                 {"role": "assistant", "content": item['output']}]} for item in data]
    dpo_data = [{"prompt": get_prompt(item), "chosen": item['output'], "rejected": ""} for item in data]
    alpacalabeled_data = [{"instruction": item['instruction'], "input": item.get('input', ''), "output": item['output'], "label": item['output']} for item in data]

    return {"ShareGPT": sharegpt_data, "ChatML": chatml_data, "OpenAI": chatml_data, "DPO": dpo_data, "Alpaca-Labeled": alpacalabeled_data}


def from_sharegpt(data):
    alpaca_data, dpo_data, alpacalabeled_data = [], [], []
    for conv in data:
        # Iterate through conversations to find human-gpt pairs
        for i in range(len(conv['conversations']) - 1):
            if conv['conversations'][i]['from'] == 'human' and conv['conversations'][i + 1]['from'] == 'gpt':
                human_msg = conv['conversations'][i]['value']
                gpt_msg = conv['conversations'][i + 1]['value']
                alpaca_data.append({"instruction": human_msg, "input": "", "output": gpt_msg})
                dpo_data.append({"prompt": human_msg, "chosen": gpt_msg, "rejected": ""})
                alpacalabeled_data.append({"instruction": human_msg, "input": "", "output": gpt_msg, "label": gpt_msg})

    chatml_data = [{"messages": [{"role": "user" if turn['from'] == 'human' else "assistant", "content": turn['value']} for turn in conv['conversations']]} for conv in data]
    return {"Alpaca": alpaca_data, "ChatML": chatml_data, "OpenAI": chatml_data, "DPO": dpo_data, "Alpaca-Labeled": alpacalabeled_data}


def from_chatml_or_openai(data):
    alpaca_data, sharegpt_data, dpo_data, alpacalabeled_data = [], [], []
    for conv in data:
        messages, system_prompt = conv.get('messages', []), ""
        if messages and messages[0]['role'] == 'system':
            system_prompt, messages = messages[0]['content'], messages[1:]

        # Create pairs for Alpaca, DPO, and Alpaca-Labeled formats
        for i in range(len(messages) - 1):
            if messages[i]['role'] == 'user' and messages[i + 1]['role'] == 'assistant':
                instruction = (f"System Prompt: {system_prompt}\n\nUser: " if system_prompt else "") + messages[i]['content']
                output = messages[i + 1]['content']
                alpaca_data.append({"instruction": instruction, "input": "", "output": output})
                dpo_data.append({"prompt": instruction, "chosen": output, "rejected": ""})
                alpacalabeled_data.append({"instruction": instruction, "input": "", "output": output, "label": output})

        # Create conversations for ShareGPT format
        conversations, first_human = [], True
        for msg in conv.get('messages', []):
            if msg['role'] == 'system': continue
            role = 'human' if msg['role'] == 'user' else 'gpt'
            content = msg['content']
            if role == 'human' and system_prompt and first_human:
                content = f"System Prompt: {system_prompt}\n\nUser: {content}"
                first_human = False
            conversations.append({"from": role, "value": content})
        if conversations:
            sharegpt_data.append({"id": str(uuid.uuid4()), "conversations": conversations})

    # The original format (ChatML or OpenAI) is a 1:1 copy
    return {
        "Alpaca": alpaca_data, "ShareGPT": sharegpt_data, "DPO": dpo_data,
        "Alpaca-Labeled": alpacalabeled_data, "ChatML": data, "OpenAI": data
    }


def from_dpo(data):
    alpaca_data = [{"instruction": item['prompt'], "input": "", "output": item['chosen']} for item in data]
    sharegpt_data = [{"id": str(uuid.uuid4()), "conversations": [
        {"from": "human", "value": item['prompt']},
        {"from": "gpt", "value": item['chosen']}]} for item in data]
    chatml_data = [{"messages": [{"role": "system", "content": "You are a helpful assistant."},
                                 {"role": "user", "content": item['prompt']},
                                 {"role": "assistant", "content": item['chosen']}]} for item in data]
    alpacalabeled_data = [{"instruction": item['prompt'], "input": "", "output": item['chosen'], "label": item['chosen']} for item in data]
    return {"Alpaca": alpaca_data, "ShareGPT": sharegpt_data, "ChatML": chatml_data, "OpenAI": chatml_data, "Alpaca-Labeled": alpacalabeled_data}

# New function for the new format
def from_alpacalabeled(data):
    # In this format, 'label' is the desired output.
    def get_prompt(item):
        return item['instruction'] + (f"\n\n{item['input']}" if item.get('input') else "")

    alpaca_data = [{"instruction": get_prompt(item), "input": "", "output": item['label']} for item in data]
    sharegpt_data = [{"id": str(uuid.uuid4()), "conversations": [
        {"from": "human", "value": get_prompt(item)},
        {"from": "gpt", "value": item['label']}]} for item in data]
    chatml_data = [{"messages": [{"role": "system", "content": "You are a helpful assistant."},
                                 {"role": "user", "content": get_prompt(item)},
                                 {"role": "assistant", "content": item['label']}]} for item in data]
    dpo_data = [{"prompt": get_prompt(item), "chosen": item['label'], "rejected": ""} for item in data]

    return {"Alpaca": alpaca_data, "ShareGPT": sharegpt_data, "ChatML": chatml_data, "OpenAI": chatml_data, "DPO": dpo_data}


# --- 3. Master Conversion and UI Logic ---

CONVERSION_MAP = {
    "Alpaca": from_alpaca,
    "ShareGPT": from_sharegpt,
    "ChatML": from_chatml_or_openai,
    "OpenAI": from_chatml_or_openai,
    "DPO": from_dpo,
    "Alpaca-Labeled": from_alpacalabeled
}

def convert_file(file, from_format, to_format):
    if not file:
        return "Please upload a file first.", None
    if from_format == to_format:
        return f"Input and output formats are the same. No conversion needed.", None

    try:
        input_data = []
        with open(file.name, "r", encoding="utf-8") as f:
            try:
                input_data = json.load(f)
            except json.JSONDecodeError:
                f.seek(0)
                input_data = [json.loads(line) for line in f if line.strip()]

        if not input_data:
             return "Could not parse any data from the file. Please ensure it's a valid JSON or JSONL file.", None

        # Get all possible conversions from the source format
        all_converted_data = CONVERSION_MAP[from_format](input_data)

        # Select the target format's data
        if to_format not in all_converted_data:
            # This case handles ChatML<->OpenAI which are identical copies
            if from_format == "ChatML" and to_format == "OpenAI":
                output_data = all_converted_data["ChatML"]
            elif from_format == "OpenAI" and to_format == "ChatML":
                output_data = all_converted_data["OpenAI"]
            else:
                 return f"Conversion from {from_format} to {to_format} is not directly supported.", None
        else:
            output_data = all_converted_data[to_format]


        # Use a temporary file to save the output as JSONL
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.jsonl', encoding='utf-8') as tf:
            for record in output_data:
                tf.write(json.dumps(record) + '\n')
            temp_path = tf.name

        return f"Successfully converted {len(input_data)} entries from {from_format} to {to_format}. Output is in JSONL format.", temp_path

    except Exception as e:
        return f"An unexpected error occurred: {str(e)}", None


def update_to_format_choices(from_choice):
    new_choices = [f for f in FORMATS if f != from_choice]
    return gr.update(choices=new_choices, value=new_choices[0])

def update_examples(from_choice):
    return gr.update(samples=[[example_paths[from_choice]]])

def load_example_to_uploader(evt: gr.SelectData):
    return gr.update(value=evt.value)


# --- 4. The Gradio Interface ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔄 Universal Dataset Converter")
    gr.Markdown("Convert between Alpaca, ShareGPT, ChatML, OpenAI, DPO, and Alpaca-Labeled formats. Select formats, upload your file, and convert.")

    with gr.Row():
        from_format = gr.Dropdown(label="Convert From", choices=FORMATS, value="Alpaca-Labeled", interactive=True)
        to_format = gr.Dropdown(label="Convert To", choices=[f for f in FORMATS if f != "Alpaca-Labeled"], value="OpenAI", interactive=True)

    with gr.Row():
        with gr.Column(scale=1):
            input_file = gr.File(label="Upload Source JSON or JSONL File", file_types=[".json", ".jsonl"])

            example_dataset = gr.Dataset(
                label="Example File (Click Row to Load)",
                components=[gr.Textbox(visible=False)],
                samples=[[example_paths["Alpaca-Labeled"]]],
                headers=["File Path"],
            )

            convert_button = gr.Button("Convert", variant="primary")

        with gr.Column(scale=2):
            status_textbox = gr.Textbox(label="Conversion Status", interactive=False)
            output_file = gr.File(label="Download Converted File (as JSONL)", interactive=False, file_types=[".jsonl"])

    # Dynamic UI updates
    from_format.change(fn=update_to_format_choices, inputs=from_format, outputs=to_format, queue=False)
    from_format.change(fn=update_examples, inputs=from_format, outputs=example_dataset, queue=False)
    example_dataset.select(fn=load_example_to_uploader, inputs=None, outputs=input_file, queue=False)

    # Button click action
    convert_button.click(
        fn=convert_file,
        inputs=[input_file, from_format, to_format],
        outputs=[status_textbox, output_file]
    )

if __name__ == "__main__":
    demo.launch()