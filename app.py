import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import os
import tempfile

st.set_page_config(page_title="DolceVita Hospitality Training Assistant", page_icon="🏨")

st.title("🏨 DolceVita Hospitality Training Assistant")
st.write("Ask anything about Luxury Hospitality Guest Experience & Leadership Skills")

if 'documents_text' not in st.session_state:
    st.session_state.documents_text = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processComplete' not in st.session_state:
    st.session_state.processComplete = False
if 'api_configured' not in st.session_state:
    st.session_state.api_configured = False

with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input("Enter Google API Key", type="password")
    uploaded_files = st.file_uploader("Upload your training documents", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
    process = st.button("Process Documents")

def extract_text_from_files(files):
    all_text = ""
    for file in files:
        file_extension = os.path.splitext(file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        try:
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)
            documents = loader.load()
            for doc in documents:
                all_text += doc.page_content + "\n\n"
        finally:
            os.remove(temp_file_path)
    return all_text

if process and uploaded_files and google_api_key:
    with st.spinner("Processing your documents..."):
        try:
            st.session_state.documents_text = extract_text_from_files(uploaded_files)
            genai.configure(api_key=google_api_key)
            st.session_state.model = genai.GenerativeModel('models/gemini-1.5-flash')
            st.session_state.processComplete = True
            st.session_state.api_configured = True
            st.success("Documents processed! You can now ask questions.")
        except Exception as e:
            st.error(f"Error: {str(e)}. Please check your API key and try again.")

if st.session_state.processComplete and st.session_state.api_configured:
    user_question = st.text_input("Ask your question:")
    if user_question:
        with st.spinner("Thinking..."):
            try:
                prompt = f"""Based on the following training materials, answer the question.

Training Materials:
{st.session_state.documents_text[:30000]}

Question: {user_question}

Answer based only on the training materials provided above. If the answer is not in the materials, say so."""
                response = st.session_state.model.generate_content(prompt)
                answer = response.text
                st.session_state.chat_history.append((user_question, answer))
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
    
    for question, answer in st.session_state.chat_history:
        st.write(f"**You:** {question}")
        st.write(f"**Assistant:** {answer}")
        st.divider()
else:
    st.info("👈 Please enter your Google API key and upload training documents to get started.")
