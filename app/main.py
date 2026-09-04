import json

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.graph import build_graph

app = FastAPI(
    title="Corrective RAG API",
    description="CRAG system using LangGraph, FAISS, Ollama and Tavily",
    version="1.0.0"
)

# Build once at startup — retriever/FAISS load happens here, not per-request
graph = build_graph()


class QueryRequest(BaseModel):
    question: str


class QueryResponse(BaseModel):
    answer: str


@app.get("/health")
def health_check():
    return {"message": "API is healthy and running!"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    result = graph.invoke({"question": req.question})
    return QueryResponse(answer=result["answer"])


async def event_stream(question: str):
    async for event in graph.astream_events(
        {"question": question},
        version="v2"
    ):
        kind = event["event"]
        node = event.get("metadata", {}).get("langgraph_node")

        # Only stream tokens from the final answer generation,
        # not from doc grading / sentence filtering / query rewriting
        if kind == "on_chat_model_stream" and node == "generate":
            content = event["data"]["chunk"].content
            if content:
                yield f"data: {json.dumps({'token': content})}\n\n"

        elif kind == "on_chain_end" and event["name"] == "LangGraph":
            yield f"data: {json.dumps({'done': True})}\n\n"


@app.post("/query/stream")
async def query_stream(req: QueryRequest):
    return StreamingResponse(
        event_stream(req.question),
        media_type="text/event-stream"
    )