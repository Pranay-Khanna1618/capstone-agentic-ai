import os
import glob
import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

md_pattern = "data/kb/*.md"

chunks = []
sources = []

vectors = []

for file_path in glob.glob(md_pattern):
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    parts = [p.strip() for p in content.split("\n\n") if p.strip()]
    chunks.extend(parts)
    sources.extend([file_path] * len(parts))


for text in chunks:
    if not text.strip():
        continue
    resp = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    vectors.append(resp.data[0].embedding)

embeddings = np.asarray(vectors, dtype="float32")

dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
faiss.write_index(index, "./index/faiss_index.bin")
np.save("./index/faiss_chunks.npy", np.array(chunks))
np.save("./index/faiss_sources.npy", np.array(sources))

print(f"Indexed chunks: {len(chunks)}")
print(f"Unique source files: {len(set(sources))}")
