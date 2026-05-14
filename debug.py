from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Load and split
loader = TextLoader("docs/sample.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)

# Embed
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings_model)

# ── 1. CHUNKS ────────────────────────────────────────────────────────────────
print("=" * 60)
print(f"CHUNKS ({len(chunks)} total)")
print("=" * 60)
for i, chunk in enumerate(chunks):
    print(f"\n[Chunk {i}]\n{chunk.page_content}")

# ── 2. RAW EMBEDDINGS ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("RAW EMBEDDINGS (first 3 chunks, first 10 dimensions shown)")
print("=" * 60)
texts = [c.page_content for c in chunks[:3]]
vectors = embeddings_model.embed_documents(texts)
for i, vec in enumerate(vectors):
    print(f"\n[Chunk {i}] vector has {len(vec)} dimensions")
    print(f"  First 10 values: {[round(v, 6) for v in vec[:10]]}")

# ── 3. SIMILARITY SEARCH WITH SCORES ─────────────────────────────────────────
print("\n" + "=" * 60)
print("SIMILARITY SEARCH WITH SCORES")
print("=" * 60)
query = "When did the Roman Empire fall?"
print(f"\nQuery: \"{query}\"\n")

results = vectorstore.similarity_search_with_score(query, k=4)
for rank, (doc, score) in enumerate(results):
    # Chroma returns L2 distance — lower = more similar
    print(f"Rank {rank + 1} | Distance: {score:.4f}")
    print(f"  {doc.page_content[:200]}")
    print()
