import streamlit as st
from tqdm import tqdm
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
    try:
        st.write("Attempting to load dataset from Hugging Face...")
        ds = datasets.load_dataset("summydev/lecturersdata", split="train")
        st.write("Dataset loaded successfully.")
    except Exception as e:
        st.warning(f"Error loading dataset from Hugging Face: {str(e)}")
        st.warning("Falling back to local dataset...")
        try:
            # Load from a local CSV file as a fallback
          #  ds = pd.read_csv("path_to_your_local_dataset.csv")  # Update with your actual path
            st.write("Local dataset loaded successfully.")
        except FileNotFoundError:
            st.error("Local dataset not found. Please provide a valid path.")
            return []

    # Process the dataset into LangchainDocument format
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
        multi_process=True,
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
                    query_vector = embedding_model.embed_query(user_query)
                    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=3)

                    # Constructing response
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
