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

