import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from tools import load_retriever, create_tools
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from memory import cached_node


class AgentState(TypedDict):
    input: str
    tool: str
    output: str

retriever = load_retriever()
tools = create_tools(retriever)

model_id = "google/flan-t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=200,
    temperature=0
)
llm = HuggingFacePipeline(pipeline=pipe)


def call_retriever(state: AgentState) -> AgentState:
    query = state["input"]

    def run(query):
        result = tools["nutrition_retriever"].invoke(query)
        if isinstance(result, list):
            return "\n".join([str(r) for r in result])
        return str(result)

    output = cached_node(query, "retriever", run)
    return {"input": query, "tool": "retriever", "output": output}


def call_llm(state: AgentState) -> AgentState:
    query = state["input"]

    def run(query):
        response = llm.invoke(query)
        if isinstance(response, list) and "generated_text" in response[0]:
            return response[0]["generated_text"]
        return str(response)

    output = cached_node(query, "llm", run)
    return {"input": query, "tool": "llm", "output": output}


# --- Router Node ---
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def router(state: AgentState):
    q = state["input"].lower()
    labels = ["nutrition", "medical", "general"]
    result = classifier(q, candidate_labels=labels)
    top_label = result["labels"][0]
    top_score = result["scores"][0]

    print(f"[Router] Top label: {top_label} (score: {top_score:.2f})")

    if top_label in ["nutrition", "medical"] and top_score > 0.5:
        return "retriever"
    else:
        return "llm"


workflow = StateGraph(AgentState)
workflow.add_node("retriever", call_retriever)
workflow.add_node("llm", call_llm)
workflow.set_entry_point("llm") 
workflow.add_conditional_edges("llm", router, {"retriever": "retriever", "llm": END})
workflow.add_edge("retriever", END)
app = workflow.compile()


# --- Examples ---
print("\n--- Example 1: Tell me about protein. ---")
print(app.invoke({"input": "Tell me about protein."}))

print("\n--- Example 2: what is 1+1? ---")
print(app.invoke({"input": "what is 1+1 ?"}))

print("\n--- Example 3: Vitamin B12 search ---")
print(app.invoke({"input": "Tell me about Vitamin B12 importance"}))
