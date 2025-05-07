import streamlit as st
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import PyPDF2
import pandas as pd
import speech_recognition as sr

# Load environment variables
load_dotenv()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "GAURAV-GPT-5.0"

# Streamlit Dark Mode
st.set_page_config(page_title="GAURAV-GPT-5.0", page_icon="🤖", layout="wide")

# Login section
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Welcome to GAURAV-GPT-5.0")
    username = st.text_input("Enter your ID / Email")
    if st.button("Login"):
        if username:
            st.session_state.authenticated = True
            st.session_state.username = username
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Please enter a valid ID to continue.")
    st.stop()

# Chat UI title
st.markdown(f"<h1 style='color:#00FFAA;'>🤖 GAURAV-GPT-5.0</h1>", unsafe_allow_html=True)

# Session history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for model settings
with st.sidebar:
    st.header("⚙️ Settings")
    llm_choice = st.radio("Choose Model Type", ["Ollama", "OpenAI"])
    model_name = st.selectbox("Model", ["mistral", "llama3", "gpt-3.5-turbo", "gpt-4"])
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    max_tokens = st.slider("Max Tokens", 50, 500, 300)
    uploaded_file = st.file_uploader("Upload a PDF or CSV", type=["pdf", "csv"])

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are GAURAV-GPT-5.0, a smart assistant. Respond clearly and helpfully."),
    ("user", "{question}")
])

# File content processing
extracted_text = ""
if uploaded_file:
    if uploaded_file.type == "application/pdf":
        reader = PyPDF2.PdfReader(uploaded_file)
        for page in reader.pages:
            extracted_text += page.extract_text()
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        extracted_text = df.to_string()

# Voice input
def recognize_speech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("🎤 Listening... Speak now.")
        audio = r.listen(source, timeout=5)
        try:
            return r.recognize_google(audio)
        except sr.UnknownValueError:
            return "Sorry, could not understand audio."
        except sr.RequestError as e:
            return f"Speech recognition error: {e}"

# Speak button
spoken_text = None
if st.sidebar.button("🎤 Use Mic"):
    spoken_text = recognize_speech()

# Generate response
def generate_response(question, model, temp, max_tok, engine):
    try:
        if engine == "Ollama":
            llm = Ollama(model=model)
        else:
            llm = ChatOpenAI(model=model, temperature=temp, max_tokens=max_tok)
        
        output_parser = StrOutputParser()
        chain = prompt | llm | output_parser
        return chain.invoke({"question": question})
    except Exception as e:
        return f"⚠️ Error: {e}"

# Chat input
user_input = st.chat_input("Type your question here...") or spoken_text

if user_input:
    final_input = user_input
    if extracted_text:
        final_input += f"\n\nAlso refer this file content:\n{extracted_text[:1500]}"

    st.chat_message("user").write(user_input)
    st.session_state.chat_history.append(("user", user_input))

    response = generate_response(final_input, model_name, temperature, max_tokens, llm_choice)
    st.chat_message("assistant").write(response)
    st.session_state.chat_history.append(("assistant", response))

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(msg)




    

