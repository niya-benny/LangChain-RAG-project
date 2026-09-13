import os

from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
    
    store = FaissVectorStore("faiss_store")

    # Load the existing index, or build and save it on the first run.
    faiss_path = os.path.join("faiss_store", "faiss.index")
    meta_path = os.path.join("faiss_store", "metadata.pkl")
    if os.path.exists(faiss_path) and os.path.exists(meta_path):
        store.load()
    else:
        docs = load_all_documents("Data")
        store.build_from_documents(docs)
    #print(store.query("What is SEO?", top_k=3))
    rag_search = RAGSearch()
    query = "What is SEO?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)