# import streamlit as st
# from tqdm import tqdm
# import pandas as pd
# from typing import List
# import datasets
# from langchain.docstore.document import Document as LangchainDocument
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from transformers import AutoTokenizer
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores.utils import DistanceStrategy

# # Load knowledge base
# @st.cache_data(show_spinner=False)
# def load_knowledge_base():
#     pd.set_option("display.max_colwidth", None)
#     try:
#         ds = datasets.load_dataset("summydev/lecturersdata", split="train")
#     except Exception as e:
#         st.error(f"Failed to load dataset: {e}")
#         return []

#     RAW_KNOWLEDGE_BASE = [
#         LangchainDocument(
#             page_content=doc["description"],
#             metadata={
#                 "source": doc["source"],
#                 "instructor_name": doc["instructorname"],
#                 "course_title": doc["coursetitle"],
#                 "rating": doc["rating"],
#             }
#         ) for doc in tqdm(ds)
#     ]

#     return RAW_KNOWLEDGE_BASE

# # Split documents
# def split_documents(knowledge_base: List[LangchainDocument], chunk_size: int, tokenizer_name: str) -> List[LangchainDocument]:
#     text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
#         AutoTokenizer.from_pretrained(tokenizer_name),
#         chunk_size=chunk_size,
#         chunk_overlap=int(chunk_size / 10),
#         add_start_index=True,
#         strip_whitespace=True,
#     )

#     docs_processed = []
#     for doc in knowledge_base:
#         docs_processed += text_splitter.split_documents([doc])

#     unique_texts = {}
#     docs_processed_unique = []
#     for doc in docs_processed:
#         if doc.page_content not in unique_texts:
#             unique_texts[doc.page_content] = True
#             docs_processed_unique.append(doc)

#     return docs_processed_unique

# # Initialize chatbot
# @st.cache_resource(show_spinner=False)
# def initialize_chatbot():
#     EMBEDDING_MODEL_NAME = "thenlper/gte-small"
#     knowledge_base = load_knowledge_base()
    
#     if not knowledge_base:
#         st.error("No knowledge base available. Exiting.")
#         return None, None

#     docs_processed = split_documents(knowledge_base, 512, EMBEDDING_MODEL_NAME)

#     embedding_model = HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL_NAME,
#         multi_process=False,  # Set to False for testing
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True},
#     )

#     KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
#         docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
#     )

#     return KNOWLEDGE_VECTOR_DATABASE, embedding_model

# # Main function
# def main():
#     st.title("Rate my PROF AI")
#     KNOWLEDGE_VECTOR_DATABASE, embedding_model = initialize_chatbot()

#     if KNOWLEDGE_VECTOR_DATABASE is None or embedding_model is None:
#         return  # Exit if initialization failed

#     user_query = st.text_input("Ask me anything about professors and courses:")

#     if st.button("Send"):
#         with st.spinner("Processing your query..."):
#             if user_query:
#                 try:
#                     query_vector = embedding_model.embed_query(user_query)
#                     retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)

#                     if retrieved_docs:
#                         st.subheader("Search Results:")
#                         for i, doc in enumerate(retrieved_docs):
#                             st.write(f"### Result {i+1}")
#                             st.write(f"**Description:** {doc.page_content}")
#                             st.write(f"**Instructor:** {doc.metadata.get('instructor_name', 'N/A')}")
#                             st.write(f"**Course Title:** {doc.metadata.get('course_title', 'N/A')}")
#                             st.write(f"**Source:** {doc.metadata.get('source', 'N/A')}")
#                     else:
#                         st.warning("No results found for your query.")
#                 except Exception as e:
#                     st.error(f"An error occurred while processing the query: {str(e)}")
#             else:
#                 st.warning("Please enter a query.")

# if __name__ == '__main__':
#     main()


import streamlit as st
from tqdm import tqdm
import pandas as pd
from typing import List
import datasets
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy

# Load knowledge base
@st.cache_resource(show_spinner=False)
def load_knowledge_base():
    pd.set_option("display.max_colwidth", None)
    ds = datasets.load_dataset("summydev/lecturersdata", split="train")

    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(
            page_content=doc["description"],
            metadata={
                "source": doc["source"],
                "instructor_name": doc["instructorname"],
                "course_title": doc["coursetitle"],
                "rating": doc["rating"],
            }
        ) for doc in tqdm(ds)
    ]

    return RAW_KNOWLEDGE_BASE

# Split documents
def split_documents(knowledge_base: List[LangchainDocument], chunk_size: int, tokenizer_name: str) -> List[LangchainDocument]:
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc)

    return docs_processed_unique

# Initialize chatbot
@st.cache_resource(show_spinner=False)
def initialize_chatbot():
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    knowledge_base = load_knowledge_base()
    
    docs_processed = split_documents(knowledge_base, 512, EMBEDDING_MODEL_NAME)

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=False,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    return KNOWLEDGE_VECTOR_DATABASE, embedding_model

# Main function
def main():
    st.title("Rate my PROF AI")
    
    # Initialize session state for messages
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    KNOWLEDGE_VECTOR_DATABASE, embedding_model = initialize_chatbot()

    # User input
    user_query = st.text_input("Ask me anything about professors and courses:")

    if st.button("Send"):
        if user_query:
            with st.spinner("Processing your query..."):
                try:
                    # System prompt to guide responses (used internally, not in the user response)
                    system_prompt = (
                        "You are a rate my professor agent to help students find classes, "
                        "that takes in user questions and answers them. "
                        "For every user question, the top 3 professors that match the user question are returned. "
                        "Use them to answer the question if needed."
                    )

                    query_vector = embedding_model.embed_query(user_query)
                    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=3)

                    # Constructing response without including the system prompt
                    response = "### Top Professors:\n"
                    if retrieved_docs:
                        for i, doc in enumerate(retrieved_docs):
                            response += f"**Professor {i+1}:** {doc.metadata.get('instructor_name', 'N/A')}\n"
                            response += f"**Course Title:** {doc.metadata.get('course_title', 'N/A')}\n"
                            response += f"**Description:** {doc.page_content}\n\n"
                    else:
                        response += "No results found for your query."

                    # Save user message and bot response
                    st.session_state.messages.append({'role': 'user', 'content': user_query})
                    st.session_state.messages.append({'role': 'bot', 'content': response})

                    # Clear input box by resetting the state
                    st.session_state['user_query'] = ""
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter a query.")

    # Display chat history
    for message in st.session_state.messages:
        if message['role'] == 'user':
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**Bot:** {message['content']}")

if __name__ == '__main__':
    main()
