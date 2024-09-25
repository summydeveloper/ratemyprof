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

class CourseChatbot:
    def __init__(self):
        pd.set_option("display.max_colwidth", None)
        self.dataset = datasets.load_dataset("summydev/lecturersdata", split="train")
        self.documents = self.create_documents()
        self.embedding_model = self.initialize_embeddings()
        self.vector_database = self.create_vector_database()
      
    def create_documents(self) -> List[LangchainDocument]:
        
 
    # Your document creation code here

        return [
            LangchainDocument(
                page_content=doc["description"],
                metadata={
                    "source": doc["source"],
                    "instructor_name": doc["instructorname"],
                    "course_title": doc["coursetitle"],
                    "rating": doc["rating"],
                }
            ) for doc in tqdm(self.dataset)
        ]

    def initialize_embeddings(self) -> HuggingFaceEmbeddings:
        embedding_model_name = "thenlper/gte-small"
        return HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            multi_process=True,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    
    def create_vector_database(self) -> FAISS:
        chunk_size = 512
        processed_docs = self.split_documents(chunk_size)
        return FAISS.from_documents(processed_docs, self.embedding_model, distance_strategy=DistanceStrategy.COSINE)

    def split_documents(self, chunk_size: int) -> List[LangchainDocument]:
        tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-small")
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=[
                "\n#{1,6} ", "\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""
            ],
        )
        
        docs_processed = []
        for doc in self.documents:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique

    def respond_to_query(self, user_query: str):
        query_vector = self.embedding_model.embed_query(user_query)
        retrieved_docs = self.vector_database.similarity_search(query=user_query, k=5)
        
        responses = []
        for i, doc in enumerate(retrieved_docs):
            responses.append({
                "rank": i + 1,
                "description": doc.page_content,
                "instructor": doc.metadata.get("instructor_name", "N/A"),
                "course_title": doc.metadata.get("course_title", "N/A"),
                "source": doc.metadata.get("source", "N/A"),
            })
        return responses

# Streamlit UI
def main():
    st.title("Course Information Chatbot")
    
    chatbot = CourseChatbot()
    
    user_input = st.text_input("What would you like to know about the courses?")
    
    if st.button("Get Info"):
        if user_input:
            responses = chatbot.respond_to_query(user_input)
            for response in responses:
                st.subheader(f"Search Result{response['rank']}")
                st.write(response['description'])
                st.write("**Instructor:**", response['instructor'])
                st.write("**Course Title:**", response['course_title'])
                st.write("**Source:**", response['source'])
                st.markdown("---")
        else:
            st.warning("Please enter a query.")

if __name__ == '__main__':
    main()
