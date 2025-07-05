from langchain.document_loaders import WebBaseLoader

def load_urls_from_file(file_path):
    with open(file_path, "r") as f:
        urls = [line.strip() for line in f if line.strip()]
    loader = WebBaseLoader(urls)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = doc.metadata.get("source", "web")
    return docs
