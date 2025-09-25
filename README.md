# MemoRAG

MemoRAG is a memory-augmented Retrieval-Augmented Generation (RAG) system that leverages LangGraph for orchestrating tool and function calls.
It integrates LLMs with specialized retrievers, intelligent routing, and semantic caching to provide fast and consistent responses for repeated or similar queries.

**Key Features**

Semantic caching for in-memory reuse of previous answers

Intelligent routing using a zero-shot classifier to select between LLM and retriever

Tool/function calling via LangGraph nodes for modular and extensible pipelines

Domain-specific knowledge retrieval (e.g., currently -- nutrition and medical queries) 

**Benefits**

Reduces redundant LLM calls, improving response time

Ensures deterministic answers for repeated or similar queries

Modular and extensible architecture allows easy integration of new tools or knowledge sources


**Getting Started**

Follow these steps to set up and run MemoRAG:

1) Clone the repository

git clone https://github.com/ShriAmogh/MemoRAG.git

cd MemoRAG

2) Download required libraries

pip install -r requirements.txt

3)Add your documents (optional)

Place PDFs or text files in the data/ folder.

Small sample documents are included for testing purposes.

4) How to run

python rag_agent.py

Then run the langgraph workflow

python main.py

5)Test Queries
Start the FastAPI server

uvicorn api:app --reload


Open the base endpoint

Postman --> http://127.0.0.1:8000/

 (GET Method)
You should see a welcome message confirming the API is running.

Submit a query (POST method)

Send a POST request to /query with your question.

{
  "query": "vitamin A ?"
}



