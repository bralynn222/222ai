import gradio as gr
import requests
import json
import time

# Global variables
conversation_history = []
available_models = ["Please enter API key and click 'Fetch Models'"]


def fetch_models(api_key):
    """Fetch available models from OpenRouter API"""
    if not api_key.strip():
        return gr.Dropdown(choices=available_models, value=available_models[0]), \
            gr.Dropdown(choices=available_models, value=available_models[0]), \
            "Please enter your API key first"

    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/models",
            headers={"Authorization": f"Bearer {api_key}"}
        )
        response.raise_for_status()
        models = response.json()["data"]
        model_ids = [model["id"] for model in models]
        model_ids.sort()

        if not model_ids:
            return gr.Dropdown(choices=available_models, value=available_models[0]), \
                gr.Dropdown(choices=available_models, value=available_models[0]), \
                "No models found for this API key"

        # Set default selections to the first two models if available
        default_model1 = model_ids[0]
        default_model2 = model_ids[1] if len(model_ids) > 1 else model_ids[0]

        return gr.Dropdown(choices=model_ids, value=default_model1), \
            gr.Dropdown(choices=model_ids, value=default_model2), \
            f"Successfully loaded {len(model_ids)} models"

    except Exception as e:
        return gr.Dropdown(choices=available_models, value=available_models[0]), \
            gr.Dropdown(choices=available_models, value=available_models[0]), \
            f"Error fetching models: {str(e)}"


def query_model(api_key, model, messages):
    """Query a model through OpenRouter API"""
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://localhost:7860",  # For tracking purposes
                "X-Title": "DPO Comparison App",
            },
            data=json.dumps({
                "model": model,
                "messages": messages
            })
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e:
        return f"Error: {str(e)}"


def chat(api_key, user_input, model1, model2, chat_history):
    """Handle chat interaction with both models"""
    if not api_key:
        return "", chat_history, "Please enter your OpenRouter API key"

    if user_input.strip() == "":
        return "", chat_history, ""

    # Add user message to history (in messages format)
    messages = [{"role": "user", "content": user_input}]

    # Get responses from both models
    response1 = query_model(api_key, model1, messages)
    response2 = query_model(api_key, model2, messages)

    # Update conversation history for DPO dataset
    conversation_history.append({
        "instruction": user_input,
        "chosen": response2,  # Model 2 as chosen
        "rejected": response1  # Model 1 as rejected
    })

    # Update chat history for display (in messages format)
    new_chat_history = chat_history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": f"Model 1 ({model1}):\n{response1}"},
        {"role": "assistant", "content": f"Model 2 ({model2}):\n{response2}"}
    ]

    return "", new_chat_history, f"Added to dataset ({len(conversation_history)} items)"


def loop_chat(api_key, user_input, model1, model2, chat_history, iterations, progress=gr.Progress()):
    """Loop chat for specified iterations with progress bar"""
    if not api_key:
        return "", chat_history, "Please enter your OpenRouter API key"

    if user_input.strip() == "":
        return "", chat_history, "Please enter a message"

    if iterations <= 0:
        return "", chat_history, "Iterations must be greater than 0"

    progress(0, desc="Starting loop...")
    new_chat_history = chat_history

    for i in range(iterations):
        progress((i + 1) / iterations, desc=f"Processing iteration {i + 1}/{iterations}")

        # Add user message to history (in messages format)
        messages = [{"role": "user", "content": user_input}]

        # Get responses from both models
        response1 = query_model(api_key, model1, messages)
        response2 = query_model(api_key, model2, messages)

        # Update conversation history for DPO dataset
        conversation_history.append({
            "instruction": user_input,
            "chosen": response2,  # Model 2 as chosen
            "rejected": response1  # Model 1 as rejected
        })

        # Update chat history for display (in messages format)
        new_chat_history = new_chat_history + [
            {"role": "user", "content": f"[Loop {i + 1}/{iterations}] {user_input}"},
            {"role": "assistant", "content": f"Model 1 ({model1}):\n{response1}"},
            {"role": "assistant", "content": f"Model 2 ({model2}):\n{response2}"}
        ]

        # Small delay to ensure progress bar updates
        time.sleep(0.1)

    return "", new_chat_history, f"Loop completed. Added {iterations} items to dataset ({len(conversation_history)} total)"


def save_dpo():
    """Save conversation history as DPO JSON file"""
    if not conversation_history:
        return "No conversations to save"

    filename = f"dpo_dataset_{int(time.time())}.json"
    with open(filename, 'w') as f:
        json.dump(conversation_history, f, indent=2)

    return f"Saved {len(conversation_history)} conversations to {filename}"


with gr.Blocks(title="DPO Comparison Tool") as demo:
    gr.Markdown("# 🤖 DPO Comparison Tool")
    gr.Markdown("Compare two models and create DPO datasets. Model 2 = Chosen, Model 1 = Rejected")

    # API Key Section
    with gr.Row():
        api_key_input = gr.Textbox(
            label="OpenRouter API Key",
            type="password",
            placeholder="Enter your OpenRouter API key"
        )
        fetch_models_btn = gr.Button("Fetch Models")

    # Model Selection
    with gr.Row():
        model1_dropdown = gr.Dropdown(
            choices=available_models,
            value=available_models[0],
            label="Model 1 (Rejected)"
        )
        model2_dropdown = gr.Dropdown(
            choices=available_models,
            value=available_models[0],
            label="Model 2 (Chosen)"
        )

    # Chat Interface
    chatbot = gr.Chatbot(label="Conversation", type="messages")
    user_input = gr.Textbox(label="Your Message", placeholder="Type your message here...")

    with gr.Row():
        iterations = gr.Number(label="Loop Iterations", value=1, precision=0, minimum=1)
        loop_btn = gr.Button("Loop")

    with gr.Row():
        send_btn = gr.Button("Send")
        save_btn = gr.Button("Save DPO Dataset")
        clear_btn = gr.Button("Clear Chat")

    status = gr.Textbox(label="Status", interactive=False)

    # Event handling
    fetch_models_btn.click(
        fetch_models,
        inputs=api_key_input,
        outputs=[model1_dropdown, model2_dropdown, status]
    )

    send_btn.click(
        chat,
        inputs=[api_key_input, user_input, model1_dropdown, model2_dropdown, chatbot],
        outputs=[user_input, chatbot, status]
    )

    user_input.submit(
        chat,
        inputs=[api_key_input, user_input, model1_dropdown, model2_dropdown, chatbot],
        outputs=[user_input, chatbot, status]
    )

    loop_btn.click(
        loop_chat,
        inputs=[api_key_input, user_input, model1_dropdown, model2_dropdown, chatbot, iterations],
        outputs=[user_input, chatbot, status]
    )

    save_btn.click(
        save_dpo,
        outputs=status
    )

    clear_btn.click(
        lambda: ([], ""),
        outputs=[chatbot, status],
        queue=False
    )

if __name__ == "__main__":
    demo.launch()