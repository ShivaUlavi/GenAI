#!/usr/bin/env python
# coding: utf-8

import os

api_key = os.getenv("AICREDITS_API_KEY")
base_url = os.getenv("AICREDITS_BASE_URL")

# Updated v1.3+ import paths
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore

# Partner packages
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# Chat model setup with extra resiliency for gateway errors
chat_model = ChatOpenAI(
    api_key=api_key,
    base_url=base_url,
    model="meta-llama/llama-3.3-70b-instruct",
    temperature=0,
    max_tokens=2048,  # Provides headroom for long regulatory citations
    max_retries=3,     # Automatically retries on 500 gateway errors
    timeout=60,        # Prevents request timeouts on long generations
    extra_body={}      # Pass custom AICredits gateway parameters here if needed
)

# Cache setup for embeddings
store = LocalFileStore("./cache/")

underlying_embeddings = OpenAIEmbeddings(
    api_key=api_key,
    base_url=base_url,
    model="text-embedding-3-large"
)

# Avoiding unnecessary costs by caching embeddings
EMBEDDINGS = CacheBackedEmbeddings.from_bytes_store(
    underlying_embeddings, store, namespace=underlying_embeddings.model
)