# RAG Pipeline
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def get_qa_chain():
    # Correct API key access from Streamlit secrets
    openai_key = st.secrets["OPENAI_API_KEY"]

    # Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_key)

    # Load FAISS index with embeddings
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Load LLM model (gpt-4 or gpt-3.5-turbo)
    llm = ChatOpenAI(model_name="gpt-4", temperature=0, openai_api_key=openai_key)

    # Create RetrievalQA chain
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain
