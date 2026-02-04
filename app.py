import streamlit as st
from huggingface_hub import InferenceClient
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

with st.sidebar:
    st.header("Configuration")
    hf_token = st.text_input("Enter Hugging Face Token", type="password")
    uploaded_files = st.file_uploader("Upload training documents", type=['pdf', 'docx', 'txt'], accept_multiple_files=True)
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

if process and uploaded_files and hf_token:
    with st.spinner("Processing documents..."):
        st.session_state.documents_text = extract_text_from_files(uploaded_files)
        st.session_state.client = InferenceClient(token=hf_token)
        st.session_state.processComplete = True
        st.success("Ready! Ask your questions.")

if st.session_state.processComplete:
    user_question = st.text_input("Ask your question:")
    if user_question:
        with st.spinner("Thinking..."):
            prompt = f"""Based on the training materials below, answer the question.

Training Materials:
{st.session_state.documents_text[:8000]}

Question: {user_question}

Answer:"""
            
            response = st.session_state.client.text_generation(
                prompt,
                model="mistralai/Mistral-7B-Instruct-v0.2",
                max_new_tokens=500
            )
            st.session_state.chat_history.append((user_question, response))
    
    for question, answer in st.session_state.chat_history:
        st.write(f"**You:** {question}")
        st.write(f"**Assistant:** {answer}")
        st.divider()
else:
    st.info("👈 Enter Hugging Face token and upload documents to start")
