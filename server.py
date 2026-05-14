from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

prompt = ChatPromptTemplate.from_template("""
Answer the question based only on the following context. Be concise.

Context:
{context}

Question: {question}
""")

chain = (
    {"context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)


class Query(BaseModel):
    question: str


@app.post("/ask")
def ask(query: Query):
    answer = chain.invoke(query.question)
    scored = vectorstore.similarity_search_with_score(query.question, k=3)
    sources = [{"text": doc.page_content, "distance": round(score, 4)} for doc, score in scored]
    return {"answer": answer, "sources": sources}


app.mount("/", StaticFiles(directory="static", html=True), name="static")
