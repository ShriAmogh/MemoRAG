import os
os.environ["USER_AGENT"] = "my-agentic-app/0.1"

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import bs4
import re
from langchain_core.documents import Document



pdf_loader = PyPDFLoader("data/Human-Nutrition-2020-Edition-1598491699.pdf")
pdf_docs = pdf_loader.load()
print(f"Loaded {len(pdf_docs)} PDF documents")


web_loader = WebBaseLoader(
    web_path="https://www.britannica.com/science/human-nutrition",
    bs_kwargs=dict(parse_only=bs4.SoupStrainer(
        class_=("topic-paragraph", "h1", "h2", "h3", "md_crosslink")
    ))
)
web_docs = web_loader.load()
print(f"Loaded {len(web_docs)} web documents")
print(web_docs[0].page_content[:100]) 


all_docs = pdf_docs + web_docs


splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,        
    chunk_overlap=100,
    separators=["\n\n", "\n", ". "]
)
chunks = splitter.split_documents(all_docs)
print(f"Total chunks created: {len(chunks)}")

def clean_text(text: str) -> str:
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text) 
    return text.strip()

cleaned_chunks = [
    Document(page_content=clean_text(doc.page_content), metadata=doc.metadata)
    for doc in chunks
    if len(doc.page_content.strip()) > 30  
]
print(f"Cleaned and retained {len(cleaned_chunks)} chunks")



#embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectordb = Chroma.from_documents(
    documents=cleaned_chunks,
    embedding=embeddings,
    persist_directory="chroma_db"
)
vectordb.persist()
print("Vector DB persisted to 'chroma_db'")


retriever = vectordb.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}  
)


query = "What are the symptoms of diabetes?"
results = retriever.invoke(query)

print("\nTop chunks for query:")
for i, res in enumerate(results):
    print(f"\n--- Chunk {i+1} ---")
    print(res.page_content[:500])
    print("Metadata:", res.metadata)
