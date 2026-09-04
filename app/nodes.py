import re
from typing import List

from pydantic import BaseModel

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langchain_tavily import TavilySearch

from app.state import State
from app.config import (
    llm,
    UPPER_TH,
    LOWER_TH
)
from app.retrieval import retriever


# ============================================================
# 1. RETRIEVAL NODE
# ============================================================

def retrieve_node(state: State) -> State:

    q = state["question"]

    retrieved = retriever.invoke(q)

    return {
        "docs": retrieved
    }


# ============================================================
# 2. DOCUMENT EVALUATION
# ============================================================

class DocEvalScore(BaseModel):
    score: float
    reason: str


doc_eval_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict retrieval evaluator for RAG.\n"
            "You will be given ONE retrieved chunk and a question.\n"
            "Return a relevance score in [0.0, 1.0].\n"
            "- 1.0: chunk alone is sufficient to answer fully/mostly\n"
            "- 0.0: chunk is irrelevant\n"
            "Be conservative with high scores.\n"
            "Also return a short reason.\n"
            "Output JSON only."
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Chunk: {chunk}"
        )
    ]
)


llm_eval_with_structured = (
    llm.with_structured_output(DocEvalScore)
)

chain_eval = (
    doc_eval_prompt
    | llm_eval_with_structured
)


def doc_eval_node(state: State) -> State:

    q = state["question"]

    scores: List[float] = []
    good: List[Document] = []

    docs = state["docs"]

    for d in docs:

        out = chain_eval.invoke(
            {
                "question": q,
                "chunk": d.page_content
            }
        )

        scores.append(out.score)

        if out.score > LOWER_TH:
            good.append(d)

    # --------------------------------------------------------
    # CORRECT
    # --------------------------------------------------------

    if any(
        score > UPPER_TH
        for score in scores
    ):

        return {
            "good_docs": good,
            "verdict": "CORRECT",
            "reason": (
                f"At least one retrieved chunk "
                f"scored > {UPPER_TH}."
            )
        }

    # --------------------------------------------------------
    # INCORRECT
    # --------------------------------------------------------

    if (
        len(scores) > 0
        and all(
            score < LOWER_TH
            for score in scores
        )
    ):

        return {
            "good_docs": [],
            "verdict": "INCORRECT",
            "reason": "No chunk was sufficient."
        }

    # --------------------------------------------------------
    # AMBIGUOUS
    # --------------------------------------------------------

    return {
        "good_docs": good,
        "verdict": "AMBIGUOUS",
        "reason": "Mixed relevance signals."
    }


# ============================================================
# 3. SENTENCE DECOMPOSITION
# ============================================================

def decompose_to_sentences(
    text: str
) -> List[str]:

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if len(sentence.strip()) > 20
    ]


# ============================================================
# 4. SENTENCE RELEVANCE FILTER
# ============================================================

class KeepOrDrop(BaseModel):

    keep: bool


filter_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a strict relevance filter.\n"
            "Return keep=true only if the sentence "
            "directly helps answer the question.\n"
            "Use ONLY the sentence.\n"
            "Output JSON only."
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Sentence:\n{sentence}"
        )
    ]
)


llm_keep = llm.with_structured_output(
    KeepOrDrop
)

chain_keep_drop = (
    filter_prompt
    | llm_keep
)


# ============================================================
# 5. REFINEMENT NODE
# ============================================================

def refine(state: State) -> State:

    q = state["question"]

    # --------------------------------------------------------
    # Use local retrieved documents
    # --------------------------------------------------------

    if state.get("verdict") == "CORRECT":

        context = "\n\n".join(
            d.page_content
            for d in state["good_docs"]
        ).strip()

    # --------------------------------------------------------
    # Use web documents
    # --------------------------------------------------------

    else:

        context = "\n\n".join(
            d.page_content
            for d in state["web_docs"]
        ).strip()

    # --------------------------------------------------------
    # Break context into sentences
    # --------------------------------------------------------

    strips = decompose_to_sentences(
        context
    )

    # --------------------------------------------------------
    # Keep only relevant sentences
    # --------------------------------------------------------

    kept: List[str] = []

    for sentence in strips:

        result = chain_keep_drop.invoke(
            {
                "question": q,
                "sentence": sentence
            }
        )

        if result.keep:
            kept.append(sentence)

    refined_context = "\n".join(
        kept
    ).strip()

    return {
        "strips": strips,
        "kept_strips": kept,
        "refined_context": refined_context
    }


# ============================================================
# 6. WEB QUERY REWRITING
# ============================================================

class WebQuery(BaseModel):

    query: str


rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Rewrite the user question into a web search "
            "query composed of keywords.\n"
            "Rules:\n"
            "- Keep it short (6–14 words).\n"
            "- If the question implies recency "
            "(e.g., recent/latest/last week/last month), "
            "add a constraint like (last 30 days).\n"
            "- Do NOT answer the question.\n"
            "- Return JSON with a single key: query."
        ),
        (
            "human",
            "Question: {question}"
        )
    ]
)


rewrite_llm = llm.with_structured_output(
    WebQuery
)

rewrite_chain = (
    rewrite_prompt
    | rewrite_llm
)


def rewrite_query_node(state: State) -> State:

    q = state["question"]

    out = rewrite_chain.invoke(
        {
            "question": q
        }
    )

    return {
        "web_query": out.query
    }


# ============================================================
# 7. WEB SEARCH NODE
# ============================================================

tavily = TavilySearch(
    max_results=5
)


def web_search_node(state: State) -> State:

    web_query = (
        state.get("web_query")
        or state["question"]
    )

    result = tavily.invoke(
        {
            "query": web_query
        }
    )

    web_docs: List[Document] = []

    for r in result or []:

        title = r.get(
            "title",
            ""
        )

        url = r.get(
            "url",
            ""
        )

        content = (
            r.get("content", "")
            or r.get("snippet", "")
        )

        text = (
            f"TITLE: {title}\n"
            f"URL: {url}\n"
            f"CONTENT:\n{content}"
        )

        web_docs.append(
            Document(
                page_content=text,
                metadata={
                    "url": url,
                    "title": title
                }
            )
        )

    return {
        "web_docs": web_docs
    }


# ============================================================
# 8. ANSWER GENERATION NODE
# ============================================================

answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful ML tutor. "
            "Answer ONLY using the provided context.\n"
            "If the context is empty or insufficient, "
            "say: 'I don't know.'"
        ),
        (
            "human",
            "Question: {question}\n\n"
            "Refined context:\n{refined_context}"
        )
    ]
)


def generate(state: State) -> State:

    output_chain = (
        answer_prompt
        | llm
    )

    output_final = output_chain.invoke(
        {
            "question": state["question"],
            "refined_context": state["refined_context"]
        }
    )

    return {
        "answer": output_final.content
    }


# ============================================================
# 9. AMBIGUOUS NODE
# ============================================================

def ambiguous_node(state: State) -> State:

    return {
        "answer": (
            f"Ambiguous: {state['reason']}"
        )
    }


# ============================================================
# 10. ROUTER
# ============================================================

def route_after_eval(state: State) -> str:

    if state["verdict"] == "CORRECT":

        return "refine"

    elif state["verdict"] == "INCORRECT":

        return "rewrite_query"

    else:

        return "ambiguous"