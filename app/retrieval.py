# app/retrieval.py

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import embedding


def load_documents():

    docs = (
        PyPDFLoader("./documents/book1.pdf").load()
        + PyPDFLoader("./documents/book2.pdf").load()
        + PyPDFLoader("./documents/book3.pdf").load()
    )

    return docs


def create_chunks(docs):

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    ).split_documents(docs)

    for d in chunks:
        d.page_content = (
            d.page_content
            .encode("utf-8", "ignore")
            .decode("utf-8", "ignore")
        )

    return chunks


def create_retriever():

    docs = load_documents()

    chunks = create_chunks(docs)

    vector_store = FAISS.from_documents(
        chunks,
        embedding
    )

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )


retriever = create_retriever()