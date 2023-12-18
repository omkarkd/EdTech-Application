import streamlit as st
from helper import get_qa_chain, create_db

st.title("CSV Q&A 🌱")
btn = st.button("Create Knowledgebase")
if btn:
    create_db()

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])





