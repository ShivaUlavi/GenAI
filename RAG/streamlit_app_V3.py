#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
from langchain_core.messages import HumanMessage

from document_loader import DocumentLoader
from rag import graph, config, retriever


# In[ ]:


st.set_page_config(page_title="Corporate Documentation Manager", layout= "wide")


# In[ ]:


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []


# In[ ]:


for message in st.session_state.chat_history:
    print(f"message: {message}")
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# In[ ]:


#docs = retriever.add_uploaded_docs(st.session_state.uploaded_files)


# In[ ]:


def process_message(message):
    """Assistant response."""
    response = graph.invoke({"messages": HumanMessage(message)}, config=config)
    return response["messages"][-1].content


# In[ ]:


st.markdown(
    """
    # Corporate Documentation Manager with Citations
    """
)


# In[ ]:


col1, col2 = st.columns([2,1])


# In[ ]:


with col1:
    st.subheader("Chat Interface")

    #React to user input
    if user_message := st.chat_input("Enter your message:"):
        #display user message in chat message container
        with st.chat_message("User"):
            st.markdown(user_message)
            #add user message to chat history
            st.session_state.chat_history.append({"role": "User", "content": user_message})
            response = process_message(user_message)
            with st.chat_message("Assistant"):
                st.markdown(response)
            #Add respinse to chat history
            st.session_state.chat_history.append(
                {"role": "Assistant", "content": response}
            )



# In[ ]:


with col2:
    st.subheader("Document Management")

    # File uploader
    uploaded_files = st.file_uploader(
        "Upload Documents",
        type=list(DocumentLoader.supported_extensions),
        accept_multiple_files=True
    )

    if uploaded_files:
        files_to_process = []
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                files_to_process.append(file)
                st.session_state.uploaded_files.append(file.name)

        # Pass the raw Streamlit UploadedFile objects directly
        if files_to_process:
            with st.spinner("Ingesting new documents into vector store..."):
                retriever.add_uploaded_docs(files_to_process)
            st.success(f"Successfully processed {len(files_to_process)} new document(s)!")

