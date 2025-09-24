from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from main import AgentState, call_retriever, call_llm, workflow as rag_workflow
from tools import load_retriever, create_tools

api = FastAPI(title="MemoRAG API", version="1.0")

retriever = None
tools = None
rag_app = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    input: str
    tool: str
    output: str

@api.on_event("startup")
def startup_event():
    global retriever, tools, rag_app

    retriever = load_retriever()
    tools = create_tools(retriever)

    rag_app = rag_workflow.compile()

@api.get("/")
def root():
    return {"message": "Welcome to MemoRAG API! Use POST /query with your input."}

@api.post("/query", response_model=QueryResponse)
def handle_query(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    state = rag_app.invoke({"input": request.query})

    return QueryResponse(
        input=state["input"],
        tool=state["tool"],
        output=state["output"]
    )
