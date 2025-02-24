import streamlit as st
import ollama
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.device_count())  # Should return the number of GPUs (should be 1 for 4090)
print(torch.cuda.get_device_name(0))

# OpenAI API key (Set in Streamlit UI)
OPENAI_API_KEY = ""

# Initialize Streamlit App
st.set_page_config(page_title="Hybrid Chatbot", page_icon="🤖", layout="wide")
st.title("💬 Hybrid AI Chatbot (Local + GPT-4)")

# Sidebar settings
st.sidebar.header("Settings")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
model_choice = st.sidebar.selectbox("Choose Local Model", ["mistral", "mixtral", "deepseek-7b", "deepseek-14b", "deepseek-coder"])
fallback_model = st.sidebar.selectbox("Choose Fallback Model", ["GPT-4 Turbo", "DeepSeek", "DeepSeek-greater"])

# Initialize OpenAI Chat Model (Only if GPT-4 is enabled)
if openai_key:
    gpt4 = ChatOpenAI(model_name="gpt-4-turbo", temperature=0.5, openai_api_key=openai_key)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# Function to run Local Model with Offloading
def local_response(prompt):
    if model_choice == "deepseek-14b":
        model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

        # Load model with offloading to RAM
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            offload_folder="offload",
            torch_dtype=torch.float16
        ).to(device)

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move input to GPU
        outputs = model.generate(**inputs, max_new_tokens=200)

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response
    elif model_choice == "deepseek-7b":
        model_name = "deepseek-r1:7b"
        response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]

    else:
        response = ollama.chat(model=model_choice, messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]


# Function to choose best model
def hybrid_chatbot(prompt):
    if fallback_model == "GPT-4 Turbo" and openai_key and ("explain" in prompt.lower() or len(prompt.split()) > 50):
        response = gpt4.stream([HumanMessage(content=prompt)])  # Enable streaming
        return "**[GPT-4 Turbo]** ", response  # Return as stream
    elif fallback_model == "DeepSeek":
        response = ollama.chat(model="deepseek-r1:7b", messages=[{"role": "user", "content": prompt}])
        return "**[DeepSeek]** ", iter([response["message"]["content"]])
    else:
        if model_choice == "deepseek-14b":
            model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                offload_folder="offload",
                torch_dtype=torch.float16
            ).to(device)

            tokenizer = AutoTokenizer.from_pretrained(model_name)

            inputs = tokenizer(prompt, return_tensors="pt").to(device)  # Move input to GPU
            outputs = model.generate(**inputs, max_new_tokens=200)

            response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return "**[DeepSeek 14B]** ", iter([response_text])  # Convert to generator

        else:
            response = ollama.chat(model=model_choice, messages=[{"role": "user", "content": prompt}])
            return "**[Local Model]** ", iter([response["message"]["content"]])  # Convert to generator

# Chat Input
user_input = st.chat_input("Type your message...")
if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response_prefix, response_generator = hybrid_chatbot(user_input)
            response_text = st.empty()  # Create empty placeholder
            full_response = response_prefix  # Start with model type tag

            for chunk in response_generator:
                full_response += chunk
                response_text.markdown(full_response)  # Update UI in real-time

        st.session_state.messages.append({"role": "assistant", "content": full_response})

    # Save response
    #st.session_state.messages.append({"role": "assistant", "content": full_response})
