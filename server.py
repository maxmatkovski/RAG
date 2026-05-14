from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA

load_dotenv()

app = FastAPI()

# Build RAG pipeline once at startup
loader = TextLoader("docs/sample.txt")
documents = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
chunks = splitter.split_documents(documents)

embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma.from_documents(chunks, embeddings_model)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask(query: Query):
    result = qa_chain.invoke({"query": query.question})

    # Get chunks with similarity scores
    scored = vectorstore.similarity_search_with_score(query.question, k=3)

    sources = []
    for doc, score in scored:
        sources.append({"text": doc.page_content, "distance": round(score, 4)})

    return {"answer": result["result"], "sources": sources}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
