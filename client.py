import streamlit as st
import requests
import pandas as pd
import io
import base64
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Healthcare Data Analysis Chatbot",
    layout="wide",
    initial_sidebar_state="expanded",
)

# CSS styles for chat messages and overall layout
st.markdown("""
<style>
/* Hide the top header and footer */
header, footer {visibility: hidden;}

/* Main container styles */
.main {
    max-width: 1200px;
    margin: 0 auto;
}

.chat-message {
    padding: 1.5rem;
    border-bottom: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 0;
    display: flex;
    align-items: flex-start;
}

.chat-message.user {
    background-color: #343541;
}

.chat-message.assistant {
    background-color: #444654;
}

.message-content {
    margin-left: 1rem;
    color: #FFFFFF;
}

/* Input container styles */
.input-container {
    position: fixed;
    bottom: 0;
    left: 0;
    right: 0;
    background-color: #343541;
    padding: 1rem;
    z-index: 1000;
}

.stTextInput input {
    background-color: #40414F;
    border-radius: 0.5rem;
    border: 1px solid rgba(255,255,255,0.1);
    padding: 0.75rem;
    color: white;
    width: 100%;
    max-width: 48rem;
    margin: 0 auto;
}

/* Add padding at the bottom to prevent content from being hidden behind input */
.main-content {
    padding-bottom: 5rem;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #202123;
}

</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'user_input' not in st.session_state:
    st.session_state.user_input = ''

def display_message(message):
    css_class = "user" if message["role"] == "user" else "assistant"
    message_content = f"""
    <div class="chat-message {css_class}">
        <div class="message-content">
            {message['content']}
        </div>
    </div>
    """
    st.markdown(message_content, unsafe_allow_html=True)

    if message["role"] == "assistant":
        if message.get("plot_image"):
            image_bytes = base64.b64decode(message["plot_image"])
            image = Image.open(io.BytesIO(image_bytes))
            st.image(image, use_column_width=True)
        if message.get("data"):
            with st.expander("Show Data"):
                df = pd.DataFrame(message["data"])
                st.dataframe(df)
        if message.get("sql_query"):
            with st.expander("Show SQL Query"):
                st.code(message["sql_query"], language='sql')

def process_input():
    user_message = st.session_state.user_input.strip()
    if user_message:
        # Add user message to chat history
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_message,
        })

        # Clear input box
        st.session_state.user_input = ''

        # Get assistant's response
        with st.spinner('Processing...'):
            try:
                # Make API request
                api_url = "http://localhost:8000/query"
                response = requests.post(api_url, json={"question": user_message})
                response.raise_for_status()
                result = response.json()

                # Assistant's message
                assistant_message = result.get('message', '')
                if not assistant_message:
                    assistant_message = "Here are the results:"

                # Add assistant's response to chat history
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": assistant_message,
                    "plot_image": result.get('plot_image'),
                    "data": result.get('data'),
                    "sql_query": result.get('sql_query'),
                })

            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# Sidebar
with st.sidebar:
    st.title("Healthcare Chatbot")
    st.markdown("### Example Questions")
    example_questions = [
        "What is the average age of patients?",
        "Show the distribution of blood pressure readings.",
        "How many patients have diabetes?",
        "Compare cholesterol levels across different age groups.",
        "What are the most common medical conditions?"
    ]
    
    for question in example_questions:
        if st.button(question):
            st.session_state.user_input = question
            process_input()

# Main chat container
with st.container():
    st.markdown('<div class="main-content">', unsafe_allow_html=True)
    for message in st.session_state.chat_history:
        display_message(message)
    st.markdown('</div>', unsafe_allow_html=True)

# Input container at the bottom
st.markdown(
    '''
    <div class="input-container">
        <div style="max-width: 48rem; margin: 0 auto;">
    ''',
    unsafe_allow_html=True
)
st.text_input("", key="user_input", on_change=process_input, placeholder="Message Healthcare Chatbot...")
st.markdown('</div></div>', unsafe_allow_html=True)
