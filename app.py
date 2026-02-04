import streamlit as st
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import os
import tempfile

st.set_page_config(page_title="DolceVita Guest Experience Training Assistant", page_icon="🏨")

st.title("🏨 DolceVita Method Training Assistant")
st.write("Ask any question about Luxury Hospitality Training and Guest Experience")

# Initialize session state
if 'conversation' not in st.session_state:
    st.session_state.conversation = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'processComplete' not in st.session_state:
    st.session_state.processComplete = None

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

def process_documents(files, api_key):
    documents = []
    
    for file in files:
        file_extension = os.path.splitext(file.name)[1]
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file.write(file.read())
            temp_file_path = temp_file.name
        
        if file_extension == ".pdf":
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == ".docx":
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == ".txt":
            loader = TextLoader(temp_file_path)
        
        documents.extend(loader.load())
        os.remove(temp_file_path)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=2
    )
