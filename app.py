import streamlit as st
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import os
import tempfile

st.set_page_config(page_title="DolceVita Training Assistant", page_icon="🏨")

st.title("🏨 DolceVita Method Training Assistant")
st.write("Ask any question about DolceVita training materials")

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
    
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True,
        output_key='answer'
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )
    
    return conversation_chain

if process and uploaded_files and google_api_key:
    with st.spinner("Processing your documents..."):
        st.session_state.conversation = process_documents(uploaded_files, google_api_key)
        st.session_state.processComplete = True
        st.success("Documents processed! You can now ask questions.")

# Chat interface
if st.session_state.processComplete:
    user_question = st.text_input("Ask a question about the DolceVita Method:")
    
    if user_question:
        with st.spinner("Thinking..."):
            response = st.session_state.conversation({'question': user_question})
            st.session_state.chat_history.append((user_question, response['answer']))
    
    # Display chat history
    for i, (question, answer) in enumerate(st.session_state.chat_history):
        st.write(f"**You:** {question}")
        st.write(f"**Assistant:** {answer}")
        st.divider()
else:
    st.info("👈 Please enter your Google API key and upload training documents to get started.")
