# import streamlit as st
# from tqdm import tqdm
# import pandas as pd
# from typing import Optional, List
# import datasets
# from langchain.docstore.document import Document as LangchainDocument
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from transformers import AutoTokenizer
# from langchain_community.vectorstores import FAISS
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain_community.vectorstores.utils import DistanceStrategy
# import torch
# from transformers import AutoModelForCausalLM, BitsAndBytesConfig, pipeline

# # Load dataset and initialize components
# def load_data():
#     ds = datasets.load_dataset("summydev/lecturersdata", split="train")
#     return [
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

# # Streamlit app
# def main():
#     st.title("RAG Chatbot")
    
#     # Load data
#     RAW_KNOWLEDGE_BASE = load_data()
    
#     # Embedding model and tokenizer
#     EMBEDDING_MODEL_NAME = "thenlper/gte-small"
#     tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
#     embedding_model = HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL_NAME,
#         multi_process=True,
#         model_kwargs={"device": "cpu"},
#         encode_kwargs={"normalize_embeddings": True},
#     )
    
#     KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
#         RAW_KNOWLEDGE_BASE, embedding_model, distance_strategy=DistanceStrategy.COSINE
#     )

#     # Reader model for generating responses
#     READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=False,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#     )
#     model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
#     reader_llm = pipeline(
#         model=model,
#         tokenizer=AutoTokenizer.from_pretrained(READER_MODEL_NAME),
#         task="text-generation",
#         do_sample=True,
#         temperature=0.2,
#         repetition_penalty=1.1,
#         return_full_text=False,
#         max_new_tokens=500,
#     )

#     # Chat input
#     user_query = st.text_input("You:", "")
#     if st.button("Send"):
#         if user_query:
#             # Perform retrieval
#             retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
#             retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
#             context = "\nExtracted documents:\n" + "\n".join([f"Document {i}:::\n{doc}" for i, doc in enumerate(retrieved_docs_text)])
            
#             # Prepare prompt
#             final_prompt = f"""Using the information contained in the context,
# provide a concise and relevant answer to the question.
# Context: {context}
# Question: {user_query}"""
            
#             # Get answer from the model
#             answer = reader_llm(final_prompt)[0]["generated_text"]
#             st.text_area("Chatbot:", value=answer, height=200)

# if __name__ == '__main__':
#     main()


import streamlit as st
from tqdm import tqdm
import pandas as pd
from typing import Optional, List
import datasets
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
import torch
from transformers import AutoModelForCausalLM, pipeline

# Load dataset and initialize components
def load_data():
    ds = datasets.load_dataset("summydev/lecturersdata", split="train")
    return [
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

# Streamlit app
def main():
    st.title("RAG Chatbot")
    
    # Load data
    RAW_KNOWLEDGE_BASE = load_data()
    
    # Embedding model and tokenizer
    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        multi_process=True,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
        RAW_KNOWLEDGE_BASE, embedding_model, distance_strategy=DistanceStrategy.COSINE
    )

    # Reader model for generating responses
    READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
    
    # Load the model without quantization
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME)
    reader_llm = pipeline(
        model=model,
        tokenizer=AutoTokenizer.from_pretrained(READER_MODEL_NAME),
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )

    # Session state for conversation history
    if 'history' not in st.session_state:
        st.session_state.history = []

    # Chat input
    user_query = st.text_input("You:", "")
    if st.button("Send"):
        if user_query:
            try:
                # Perform retrieval
                retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
                retrieved_docs_text = [doc.page_content for doc in retrieved_docs]
                context = "\nExtracted documents:\n" + "\n".join([f"Document {i}:::\n{doc}" for i, doc in enumerate(retrieved_docs_text)])
                
                # Prepare prompt
                final_prompt = f"""Using the information contained in the context,
provide a concise and relevant answer to the question.
Context: {context}
Question: {user_query}"""
                
                # Get answer from the model
                answer = reader_llm(final_prompt)[0]["generated_text"]

                # Store the conversation in session state
                st.session_state.history.append({"user": user_query, "bot": answer})
                
            except Exception as e:
                st.error(f"Error: {e}")

    # Display conversation history
    if st.session_state.history:
        st.write("### Conversation History")
        for chat in st.session_state.history:
            st.write(f"You: {chat['user']}")
            st.write(f"Bot: {chat['bot']}")
            st.write("---")

if __name__ == '__main__':
    main()
