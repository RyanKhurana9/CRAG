from langgraph.graph import (
    START,
    END,
    StateGraph
)

from app.state import State

from app.nodes import (
    retrieve_node,
    doc_eval_node,
    refine,
    rewrite_query_node,
    web_search_node,
    generate,
    ambiguous_node,
    route_after_eval
)


def build_graph():

    graph = StateGraph(State)

    # -----------------------------
    # Nodes
    # -----------------------------

    graph.add_node(
        "retrieve_node",
        retrieve_node
    )

    graph.add_node(
        "doc_eval_node",
        doc_eval_node
    )

    graph.add_node(
        "refine",
        refine
    )

    graph.add_node(
        "rewrite_query_node",
        rewrite_query_node
    )

    graph.add_node(
        "web_search_node",
        web_search_node
    )

    graph.add_node(
        "generate",
        generate
    )

    graph.add_node(
        "ambiguous_node",
        ambiguous_node
    )

    # -----------------------------
    # Starting edges
    # -----------------------------

    graph.add_edge(
        START,
        "retrieve_node"
    )

    graph.add_edge(
        "retrieve_node",
        "doc_eval_node"
    )

    # -----------------------------
    # Conditional routing
    # -----------------------------

    graph.add_conditional_edges(
        "doc_eval_node",
        route_after_eval,
        {
            "refine": "refine",
            "rewrite_query": "rewrite_query_node",
            "ambiguous": "ambiguous_node"
        }
    )

    # -----------------------------
    # Web path
    # -----------------------------

    graph.add_edge(
        "rewrite_query_node",
        "web_search_node"
    )

    graph.add_edge(
        "web_search_node",
        "refine"
    )

    # -----------------------------
    # Final path
    # -----------------------------

    graph.add_edge(
        "refine",
        "generate"
    )

    graph.add_edge(
        "generate",
        END
    )

    graph.add_edge(
        "ambiguous_node",
        END
    )

    return graph.compile()