# word_loader.py
from langchain.document_loaders import Docx2txtLoader

def load_word(path):
    loader = Docx2txtLoader(path)
    return loader.load()
