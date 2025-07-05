# rag_chain.py

from langchain.vectorstores import FAISS
from langchain.embeddings import OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.llms import Ollama

def load_vectorstore(path="data/vectorstore"):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)




def get_rag_chain(source_filter=None):
    vectorstore = load_vectorstore()

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5, "filter": build_filter(source_filter)}
    )

    llm = Ollama(model="llama3")
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )

def build_filter(source_list):
    if not source_list:
        return None
    return {"source": {"$in": source_list}}

