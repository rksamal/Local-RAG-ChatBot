# chatbot_ui.py


import sys
import os

# Add the parent directory of 'app' to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag_chain import get_rag_chain
import streamlit as st
from app.rag_chain import get_rag_chain
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
import tempfile
import os

from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

embeddings = OllamaEmbeddings(model="nomic-embed-text")
llm = Ollama(model="llama3")


# ----------------------------
# UI Setup
# ----------------------------
st.set_page_config(page_title="Cotiviti RAG Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Cotiviti Knowledge Assistant")
st.markdown("Ask about **CMS** or **NCQA** specs, HEDIS rules, quality measures, or domain concepts.")

# ----------------------------
# Sidebar Filters
# ----------------------------
with st.sidebar:
    selected_sources = st.multiselect(
        "📁 Filter by Source",
        options=["CMS", "NCQA", "Other"],
        default=["CMS", "NCQA"]
    )

    uploaded_files = st.file_uploader(
        "📤 Upload documents (optional)",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )

# ----------------------------
# Session State Setup
# ----------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

# ----------------------------
# Use Uploaded Files or VectorDB
# ----------------------------
def build_chain_from_uploads(files):
    docs = []
    for file in files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.name)[1]) as tmp_file:
            tmp_file.write(file.read())
            tmp_path = tmp_file.name

        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(tmp_path)
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(tmp_path)
        elif file.name.endswith(".txt"):
            loader = TextLoader(tmp_path)
        else:
            continue

        file_docs = loader.load()
        for doc in file_docs:
            doc.metadata["source"] = file.name
        docs.extend(file_docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vectordb = FAISS.from_documents(chunks, embeddings)

    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    memory = ConversationBufferMemory(return_messages=True)
    llm = Ollama(model="llama3")

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    ), retriever

# Create chain (uploaded vs persistent vectorstore)
if uploaded_files:
    st.session_state.qa_chain, custom_retriever = build_chain_from_uploads(uploaded_files)
else:
    st.session_state.qa_chain = get_rag_chain(selected_sources)
    custom_retriever = st.session_state.qa_chain.retriever

# ----------------------------
# Chat Input Box
# ----------------------------
with st.form("chat_form"):
    user_query = st.text_input("💬 Your question:", placeholder="e.g., What is HEDIS?")
    submit = st.form_submit_button("Ask")

if submit and user_query:
    response = st.session_state.qa_chain.run(user_query)
    st.session_state.chat_history.append(("You", user_query))
    st.session_state.chat_history.append(("Bot", response))

    # Get sources used
    with st.expander("🔎 Sources used"):
        relevant_docs = custom_retriever.get_relevant_documents(user_query)
        for doc in relevant_docs:
            st.markdown(f"- **{doc.metadata.get('source', 'Unknown')}**")

# ----------------------------
# Chat History Display
# ----------------------------
for role, message in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(message)
