# RAG Pipeline
import streamlit as st
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

def get_qa_chain():
    # Set the OpenAI API key from Streamlit secrets
    import openai
    openai.api_key = st.secrets["openai"]["api_key"]

    # Load FAISS index and setup the retriever
    db = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Setup the language model
    llm = ChatOpenAI(model_name="gpt-4", temperature=0)

    # Create and return the QA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
