# pdf_loader.py
from langchain.document_loaders import PyPDFLoader

def load_pdf(path):
    loader = PyPDFLoader(path)
    return loader.load()
