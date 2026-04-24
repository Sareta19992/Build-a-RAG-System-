import streamlit as st
from rag_pipeline import retrieve_documents, generate_simple_answer

st.set_page_config(page_title="RAG Assignment 4", page_icon="🤖")

st.title("RAG System Demo")
st.write("Ask a question about CNN, RNN, or Transformer models.")

query = st.text_input("Enter your question:")

if query:
    try:
        docs, context = retrieve_documents(query)
        answer = generate_simple_answer(query, context)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Chunks")
        for i, doc in enumerate(docs, start=1):
            st.write(f"**Chunk {i}:**")
            st.write("---")
            st.write(doc.page_content)

    except Exception as e:
        st.error(f"Error: {e}")