import streamlit as st
from main import process_request
import json

# Set page configuration for a more professional look
st.set_page_config(page_title="Customer Service Knowledge Assistant", layout="wide")

# Custom CSS for premium styling
st.markdown(
    """
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .response-container {
        padding: 2.5rem;
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 2rem;
    }
    .retrieved-container {
        padding: 1.5rem;
        background-color: #e9ecef;
        border-radius: 10px;
        border: 1px solid #dee2e6;
    }
    h1 {
        color: #1e293b;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    h2 {
        color: #334155;
        font-family: 'Inter', sans-serif;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Search & Retrieval Demo")
st.markdown("---")

# Layout with columns
col1, col2 = st.columns([2, 1])

with col1:
    query = st.text_input(
        "Customer Query", placeholder="e.g., My tracking number shows no movement..."
    )

with col2:
    message_id = st.text_input("Thread/Message ID", value="default_user")

if st.button("Query Knowledge Base"):
    if query and message_id:
        with st.spinner("Analyzing knowledge base..."):
            try:
                raw_response = process_request(query, message_id)
                data = json.loads(raw_response)

                answer = data.get("response", "No answer formulated.")
                retrieved_docs = data.get(
                    "retrieved_data_used", "No documents retrieved."
                )

                # Answer Box
                st.subheader("Answer")
                st.info(answer)

                # Retrieved Documents Box
                st.subheader("Retrieved Documents")
                if isinstance(retrieved_docs, list) and retrieved_docs:
                    for i, doc in enumerate(retrieved_docs):
                        with st.container():
                            st.markdown(
                                f"**Document {i + 1}** (Score: {doc.get('score', 0):.4f})"
                            )
                            st.info(doc.get("content", ""))

                            # Display Metadata in a small code block or sub-text
                            with st.expander(f"Metadata for Doc {i + 1}"):
                                st.json(doc.get("metadata", {}))
                            st.markdown("---")
                elif isinstance(retrieved_docs, str) and retrieved_docs:
                    st.markdown(retrieved_docs)
                else:
                    st.write("No source documents found for this query.")

            except json.JSONDecodeError:
                st.error("Failed to parse response from system.")
                st.write(raw_response)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please enter both a query and a message ID to continue.")
