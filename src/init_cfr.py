import os
import json

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

if __name__ == "__main__":
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                    model_kwargs={"device": "cpu"},
                                    encode_kwargs={'normalize_embeddings': True})

    # Read data from the JSON file
    with open('data/litfl.json', 'r', encoding='utf-8') as json_file:
        litfl_data = json.load(json_file)
    
    chroma_dir = 'data/chroma_db'
    if not os.path.exists(chroma_dir):
        print("Createing ChromaDB")
        documents = [Document(page_content=key, metadata={"source": i}) 
                            for i, key in enumerate(litfl_data.keys())]
        documents = Chroma.from_documents(documents, 
                                            embeddings, 
                                            persist_directory=chroma_dir, 
                                            collection_name='LITFL')
    else:
        print("ChromaDB has already been initialize")