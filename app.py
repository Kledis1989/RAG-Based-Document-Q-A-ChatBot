import streamlit as st
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.chains import RetrievalQA
import os
import tempfile

st.set_page_config(page_title="DolceVita Training Assistant", page_icon="🏨")

st.title("🏨 DolceVita Method Training Assistant")
st.write("Ask any question about DolceVita training materials")

# Initialize session state
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
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
        chunk_overlap=200
    )
    doc_chunks = text_splitter.split_documents(documents)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key
    )
    
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    
    llm = GoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=api_key,
        temperature=0.3
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    return qa_chain

if process and uploaded_files and google_api_key:
    with st.spinner("Processing your documents..."):
        st.session_state.qa_chain = process_documents(uploaded_files, google_api_key)
        st.session_state.processComplete = True
        st.success("Documents processed! You can now ask questions.")

# Chat interface
if st.session_state.processComplete:
    user_question = st.text_input("Ask a question about the DolceVita Method:")
    
    if user_question:
        with st.spinner("Thinking..."):
            response = st.session_state.qa_chain({"query": user_question})
            answer = response['result']
            st.session_state.chat_history.append((user_question, answer))
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        st.write(f"**You:** {question}")
        st.write(f"**Assistant:** {answer}")
        st.divider()
else:
    st.info("👈 Please enter your Google API key and upload training documents to get started.")
