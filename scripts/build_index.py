import os
import glob
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

md_pattern = "data/kb/*.md"

chunks = []
sources = []
documents = []

# Load and chunk markdown files
for file_path in glob.glob(md_pattern):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    parts = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks.extend(parts)
    sources.extend([file_path] * len(parts))
    
    # Create Document objects for FAISS
    for i, part in enumerate(parts):
        doc = Document(
            page_content=part,
            metadata={"source": file_path, "chunk": i}
        )
        documents.append(doc)

if not documents:
    print("No documents found. Check your data/kb/ folder.")
    exit(1)

print(f"Total chunks to embed: {len(documents)}")

# Create embeddings and FAISS index using LangChain
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(documents, embeddings)

# Save the index
index_dir = "./index"
os.makedirs(index_dir, exist_ok=True)
vector_store.save_local(index_dir)

# Also save numpy arrays for reference
np.save(os.path.join(index_dir, "faiss_chunks.npy"), np.array(chunks))
np.save(os.path.join(index_dir, "faiss_sources.npy"), np.array(sources))

print(f"Indexed chunks: {len(chunks)}")
print(f"Unique source files: {len(set(sources))}")
print(f"Index saved to {index_dir}")