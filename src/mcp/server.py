from typing import Dict, Any
from src.memory.vector_memory import VectorMemory
from jsonrpcserver import method, serve

vm = VectorMemory("analysis_agent")

@method
def get_deal_insights(deal_text: str) -> Dict[str, Any]:
    retriever = vm.retriever(k=5)
    docs = retriever.invoke(deal_text)
    insights = [doc.page_content for doc in docs]
    return {"insights": insights}

if __name__ == "__main__":
    serve() # starts a server on localhost:5000 by default