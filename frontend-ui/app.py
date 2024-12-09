import streamlit as st
import sys
import os, json

# Add the src directory to the path to import perform_rag
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src', 'perform_rag')))
from perform_rag import main as perform_rag_main

print("helo")
#st.write(st.secrets)

# Streamlit UI
st.title("Global Tech Colab For Good: A Platform for Non-Profits and Research Groups")
st.write("Enter a problem statement to find relevant tech research papers and get an explanation for bonus!")

# User query input
query = st.text_input("Enter your query:", "")

if st.button("Submit"):
    with st.spinner("Fetching relevant papers and generating explanation..."):
        try:
            # Get the answer from the RAG pipeline
            answer = perform_rag_main(query)
            st.success("Explanation generated successfully!")
            st.write(answer)
        except Exception as e:
            st.error(f"An error occurred: {e}")
