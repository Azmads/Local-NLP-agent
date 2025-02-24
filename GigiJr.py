import streamlit as st
import ollama
import openai
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage


# OpenAI API key (Set in Streamlit UI)
OPENAI_API_KEY = ""

# Initialize Streamlit App
st.set_page_config(page_title="Hybrid Chatbot", page_icon="🤖", layout="wide")
st.title("💬 Hybrid AI Chatbot (Local + GPT-4)")

# Sidebar settings
st.sidebar.header("Settings")
openai_key = st.sidebar.text_input("OpenAI API Key", type="password")
model_choice = st.sidebar.selectbox("Choose Local Model", ["mistral", "mixtral", "deepseek-7b", "deepseek-14b", "deepseek-coder"])
fallback_model = st.sidebar.selectbox("Choose Fallback Model", ["GPT-4 Turbo", "DeepSeek"])

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


# Function to run Local Model
def local_response(prompt):
    response = ollama.chat(model=model_choice, messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]


# Function to choose best model
def hybrid_chatbot(prompt):
    if fallback_model == "GPT-4 Turbo" and openai_key and ("explain" in prompt.lower() or len(prompt.split()) > 50):
        response = gpt4([HumanMessage(content=prompt)])
        return f"**[GPT-4 Turbo]** {response.content}"
    elif fallback_model == "DeepSeek":
        response = ollama.chat(model="deepseek-7b", messages=[{"role": "user", "content": prompt}])
        return f"**[DeepSeek]** {response['message']['content']}"
    else:
        response = local_response(prompt)
        return f"**[Local Model]** {response}"

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
            response = hybrid_chatbot(user_input)
            st.markdown(response)

    # Save response
    st.session_state.messages.append({"role": "assistant", "content": response})
