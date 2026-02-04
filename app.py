import streamlit as st
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import os
import tempfile

st.set_page_config(page_title="DolceVita Training Assistant", page_icon="🏨")

st.title("🏨 DolceVita Method Training Assistant")
st.write("Ask any question about Luxury Hospitality Guest Experience or Leadership skills")

# Initialize session state
if 'documents_text' not in st.session_state:
    st.session_state.documents_text = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processComplete' not in st.session_state:
    st.session_state.processComplete = False

# Sidebar for API key and file upload
with st.sidebar:
    st.header("Configuration")
    google_api_key = st.text_input("Enter Google API Key", type="password")
    
    uploaded_files = st.file_uploader(
        "Upload your training documents",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True
    )
    
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
        st.session_state.documents_text = extract_text_from_files(uploaded_files)
        st.session_state.processComplete = True
        genai.configure(api_key=google_api_key)
        st.session_state.model = genai.GenerativeModel('gemini-1.5-flash')
        st.success("Documents processed! You can now ask questions.")

# Chat interface
if st.session_state.processComplete:
    user_question = st.text_input("Ask a question about Luxury Hospitality Guest Experience or Leadership Skills:")
    
    if user_question:
        with st.spinner("Thinking..."):
            prompt = f"""Based on the following training materials, answer the question.

Training Materials:
{st.session_state.documents_text[:30000]}

Question: {user_question}

Answer based only on the training materials provided above. If the answer is not in the materials, say so."""
            
            response = st.session_state.model.generate_content(prompt)
            answer = response.text
            st.session_state.chat_history.append((user_question, answer))
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_s
