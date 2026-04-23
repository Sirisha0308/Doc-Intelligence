from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import shutil
import os

# ── Clear old database ───────────────────────────────
if os.path.exists("./chroma_db"):
    shutil.rmtree("./chroma_db")
    print("Old database cleared")

# ── Stage 2: Load documents ──────────────────────────
print("Stage 2: Loading documents...")
loader = DirectoryLoader(
    "documents/",
    glob="**/*.pdf",
    loader_cls=PyMuPDFLoader
)
docs = loader.load()
print(f"         {len(docs)} pages loaded")

# ── Stage 3: Chunking ────────────────────────────────
print("Stage 3: Splitting into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_documents(docs)
print(f"         {len(chunks)} chunks created")

# ── Stage 4: Embedding + Vector Store ───────────────
print("Stage 4: Loading embedding model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
print("         Embedding model ready!")
print("         Converting chunks to vectors...")
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
print(f"         Done! {vectorstore._collection.count()} vectors stored")
print("")

# ── Stage 5: LLM + Retrieval Chain ──────────────────
print("Stage 5: Loading LLM...")
llm = OllamaLLM(model="llama3.2")
print("         LLM ready!")

# retriever fetches top 3 relevant chunks
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# prompt template
prompt = PromptTemplate(
    template="""You are a helpful assistant. Use only the
context below to answer the question. If the answer is not
in the context, say "I don't know based on the documents."

Context:
{context}

Question: {question}

Answer:""",
    input_variables=["context", "question"]
)

# helper to format retrieved chunks into one string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# build the RAG chain using new LangChain syntax
rag_chain = (
    {"context": retriever | format_docs,
     "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

print("         RAG chain ready!")
print("")

# ── Test with 3 questions ────────────────────────────
print("=" * 50)
print("DOCUMENT INTELLIGENCE SYSTEM")
print("=" * 50)
print("")

test_questions = [
    "what is attention mechanism?",
    "what is retrieval augmented generation?",
    "what are large language models?"
]

for question in test_questions:
    print(f"Question: {question}")
    print("")

    answer = rag_chain.invoke(question)

    print(f"Answer: {answer}")
    print("")

    # show which chunks were used
    source_docs = retriever.invoke(question)
    print("Sources used:")
    seen_sources = set()
    for doc in source_docs:
        source = doc.metadata["source"]
        page = doc.metadata.get("page", "N/A")
        if source not in seen_sources:
            print(f"  - {source} (page {page})")
            seen_sources.add(source)
    print("")
    print("-" * 50)
    print("")

print("Stage 5 complete!")