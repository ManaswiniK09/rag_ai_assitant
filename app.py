import streamlit as st
from qa_chain import get_qa_chain

st.set_page_config(page_title="Company AI Assistant")
st.title("Ask the AI assistant")

question = st.text_input("Ask the question")
if question:
    qa_chain = get_qa_chain()
    response = qa_chain.run(question)
    st.write(response)


