import streamlit as st
from main import process_request

st.title("Query Processor")

query = st.text_input("Enter your query:")
message_id = st.text_input("Enter message ID:")

if st.button("Process"):
    if query and message_id:
        with st.spinner("Processing..."):
            response =  process_request(query, message_id)
        st.success("Response:")
        st.write(response)
    else:
        st.error("Please fill in both the query and message ID.")