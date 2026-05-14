from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

# 1. Load documents
loader = TextLoader("docs/sample.txt")
documents = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)

print(f"Loaded {len(documents)} document(s), split into {len(chunks)} chunks.")

# 3. Embed and store in ChromaDB
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 4. Set up retriever + LLM
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 5. Build RAG chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

# 6. Ask questions
print("\nRAG ready! Ask anything about Roman history (or type 'quit' to exit).\n")
while True:
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        break
    if not question:
        continue

    result = qa_chain.invoke({"query": question})
    print(f"\nAnswer: {result['result']}")
    print("\nSources used:")
    for doc in result["source_documents"]:
        print(f"  - ...{doc.page_content[:120]}...")
    print()
