import streamlit as st
from qa_chain import get_qa_chain

# App title
st.set_page_config(page_title="🧠 RAG AI Assistant", layout="wide")
st.title("📄 Ask Your PDF")
st.caption("Powered by OpenAI + LangChain + FAISS")

# Initialize the QA chain once
@st.cache_resource
def load_chain():
    return get_qa_chain()

qa_chain = load_chain()

# Text input
question = st.text_input("Ask a question based on your uploaded document:", "")

if question:
    with st.spinner("Thinking... 🤔"):
        try:
            response = qa_chain.run(question)
            st.success(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
