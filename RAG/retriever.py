#!/usr/bin/env python
# coding: utf-8

import os
import tempfile
from typing import List, Any

import streamlit as st
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

from document_loader import load_document
from llms import EMBEDDINGS


def get_vector_store() -> InMemoryVectorStore:
    """Retrieve or initialize the vector store in Streamlit session state."""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = InMemoryVectorStore(embedding=EMBEDDINGS)
    return st.session_state.vector_store


def split_documents(docs: List[Document]) -> List[Document]:
    """Split each document."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500, chunk_overlap=200
    )
    return text_splitter.split_documents(docs)


class DocumentRetriever(BaseRetriever):
    """A retriever that contains the top k documents that contain the user query."""
    documents: List[Document] = []
    k: int = 5

    def model_post_init(self, ctx: Any) -> None:
        self.store_documents(self.documents)

    @staticmethod
    def store_documents(docs: List[Document]) -> None:
        """Add documents to the vector store."""
        splits = split_documents(docs)
        vector_store = get_vector_store()
        vector_store.add_documents(splits)

    def add_uploaded_docs(self, uploaded_files):
        """Add uploaded documents."""
        docs = []
        temp_dir = tempfile.TemporaryDirectory()
        for file in uploaded_files:
            temp_filepath = os.path.join(temp_dir.name, file.name)
            with open(temp_filepath, "wb") as f:
                f.write(file.getvalue())
                docs.extend(load_document(temp_filepath))
        self.documents.extend(docs)
        self.store_documents(docs)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        """Sync implementations for retriever."""
        if len(self.documents) == 0:
            return []
        vector_store = get_vector_store()
        return vector_store.similarity_search(query=query, k=self.k)