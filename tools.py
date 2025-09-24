from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.agents import tool

def load_retriever():
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )
    return vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})


def create_tools(retriever):
    @tool
    def nutrition_retriever(query: str) -> str:
        """Retrieve nutrition-related information from the vector database."""
        docs = retriever.invoke(query)
        return "\n\n".join([doc.page_content[:300] for doc in docs])

    return {"nutrition_retriever": nutrition_retriever}
