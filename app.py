# from flask import Flask, request, jsonify, render_template
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

# app = Flask(__name__)

# def load_knowledge_base():
#     pd.set_option("display.max_colwidth", None)
#     ds = datasets.load_dataset("summydev/lecturersdata", split="train")

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

# def initialize_chatbot():
#     EMBEDDING_MODEL_NAME = "thenlper/gte-small"
#     knowledge_base = load_knowledge_base()
#     docs_processed = split_documents(knowledge_base, 512, EMBEDDING_MODEL_NAME)

#     embedding_model = HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL_NAME,
#         multi_process=True,
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True},
#     )

#     KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
#         docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
#     )

#     return KNOWLEDGE_VECTOR_DATABASE, embedding_model

# KNOWLEDGE_VECTOR_DATABASE, embedding_model = initialize_chatbot()

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/ask', methods=['POST'])
# def ask():
#     user_query = request.form['query']
#     query_vector = embedding_model.embed_query(user_query)
#     retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    
#     results = []
#     for doc in retrieved_docs:
#         results.append({
#             "description": doc.page_content,
#             "instructor": doc.metadata.get("instructor_name", "N/A"),
#             "course_title": doc.metadata.get("course_title", "N/A"),
#             "source": doc.metadata.get("source", "N/A"),
#         })
    
#     return jsonify(results)

# if __name__ == '__main__':
#     app.run(debug=True)

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

# # Load knowledge base and initialize the chatbot
# def load_knowledge_base():
#     pd.set_option("display.max_colwidth", None)
#     ds = datasets.load_dataset("summydev/lecturersdata", split="train")

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

# def initialize_chatbot():
#     EMBEDDING_MODEL_NAME = "thenlper/gte-small"
#     print("Loading knowledge base...")
#     knowledge_base = load_knowledge_base()
#     print("Knowledge base loaded.")
    
#     print("Splitting documents...")
#     docs_processed = split_documents(knowledge_base, 512, EMBEDDING_MODEL_NAME)
#     print(f"Documents processed: {len(docs_processed)}")

#     print("Initializing embedding model...")
#     embedding_model = HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL_NAME,
#         multi_process=True,
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True},
#     )

#     print("Creating knowledge vector database...")
#     KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
#         docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
#     )
#     print("Knowledge vector database created.")

#     return KNOWLEDGE_VECTOR_DATABASE, embedding_model

# def main():
#     global KNOWLEDGE_VECTOR_DATABASE, embedding_model
#     KNOWLEDGE_VECTOR_DATABASE, embedding_model = initialize_chatbot()

#     # Streamlit UI
#     st.title("Course & Instructor Chatbot")
#     user_query = st.text_input("Ask me anything about professors and courses:")

#     if st.button("Send"):
#         st.write("Query received, processing...")  # Feedback to user
#         if user_query:
#             st.write(f"User query: {user_query}")  # Debugging statement
#             try:
#                 query_vector = embedding_model.embed_query(user_query)
#                 st.write("Query vector generated.")  # Debugging statement
#                 retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
#                 st.write("Documents retrieved.")  # Debugging statement

#                 if retrieved_docs:
#                     st.subheader("Search Results:")
#                     for i, doc in enumerate(retrieved_docs):
#                         st.write(f"### Result {i+1}")
#                         st.write(f"**Description:** {doc.page_content}")
#                         st.write(f"**Instructor:** {doc.metadata.get('instructor_name', 'N/A')}")
#                         st.write(f"**Course Title:** {doc.metadata.get('course_title', 'N/A')}")
#                         st.write(f"**Source:** {doc.metadata.get('source', 'N/A')}")
#                 else:
#                     st.warning("No results found for your query.")
#             except Exception as e:
#                 st.error(f"An error occurred: {str(e)}")  # Show any errors
#         else:
#             st.warning("Please enter a query.")

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
@st.cache_data(show_spinner=False)
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
        multi_process=False,  # Set to False for testing
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    return KNOWLEDGE_VECTOR_DATABASE, embedding_model

# Main function
def main():
    global KNOWLEDGE_VECTOR_DATABASE, embedding_model
    KNOWLEDGE_VECTOR_DATABASE, embedding_model = initialize_chatbot()

    # Streamlit UI
    st.title("Course & Instructor Chatbot")
    user_query = st.text_input("Ask me anything about professors and courses:")

    if st.button("Send"):
        with st.spinner("Processing your query..."):
            if user_query:
                try:
                    query_vector = embedding_model.embed_query(user_query)
                    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)

                    if retrieved_docs:
                        st.subheader("Search Results:")
                        for i, doc in enumerate(retrieved_docs):
                            st.write(f"### Result {i+1}")
                            st.write(f"**Description:** {doc.page_content}")
                            st.write(f"**Instructor:** {doc.metadata.get('instructor_name', 'N/A')}")
                            st.write(f"**Course Title:** {doc.metadata.get('course_title', 'N/A')}")
                            st.write(f"**Source:** {doc.metadata.get('source', 'N/A')}")
                    else:
                        st.warning("No results found for your query.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")  # Show any errors
            else:
                st.warning("Please enter a query.")

if __name__ == '__main__':
    main()
