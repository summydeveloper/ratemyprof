# from tqdm import tqdm
# import pandas as pd
# from typing import Optional, List
# from datasets import load_dataset
# import matplotlib.pyplot as plt
# from langchain.docstore.document import Document as LangchainDocument
# from langchain.text_splitter import RecursiveCharacterTextSplitter
# from transformers import AutoTokenizer

# # Load dataset
# ds = load_dataset("summydev/lecturersdata", split="train")
# print(ds.column_names)  # Show column names

# # Check the structure of the first few records
# print(ds[:5])  # Print the first few records to see their structure

# # Initialize RAW_KNOWLEDGE_BASE
# RAW_KNOWLEDGE_BASE = []
 
 

# for doc in tqdm(ds):
#     try:
#         # Adjust based on the actual column names in the dataset
#         raw_doc =   LangchainDocument(
#         page_content=f"{doc['__index_level_0__']} - {doc['__index_level_1__']} - {doc['text']}", 
#         metadata={"source": doc["source"]}
#     )  
#         RAW_KNOWLEDGE_BASE.append(raw_doc)
#     except KeyError as e:
#         print(f"KeyError: {e}. Document: {doc}")

# EMBEDDING_MODEL_NAME = "thenlper/gte-small"
# MARKDOWN_SEPARATORS = [
#     "\n#{1,6} ",
#     "```\n",
#     "\n\\*\\*\\*+\n",
#     "\n---+\n",
#     "\n___+\n",
#     "\n\n",
#     "\n",
#     " ",
#     "",
# ]


# def split_documents(
#     chunk_size: int,
#     knowledge_base: List[LangchainDocument],
#     tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
# ) -> List[LangchainDocument]:
#     text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
#         AutoTokenizer.from_pretrained(tokenizer_name),
#         chunk_size=chunk_size,
#         chunk_overlap=int(chunk_size / 10),
#         add_start_index=True,
#         strip_whitespace=True,
#         separators=MARKDOWN_SEPARATORS,
#     )
    
#     docs_processed = []
#     for doc in knowledge_base:
#         docs_processed += text_splitter.split_documents([doc])

#     # Remove duplicates
#     unique_texts = {}
#     docs_processed_unique = []
#     for doc in docs_processed:
#         if doc.page_content not in unique_texts:
#             unique_texts[doc.page_content] = True
#             docs_processed_unique.append(doc)

#     return docs_processed_unique

# docs_processed = split_documents(
#     512,
#     RAW_KNOWLEDGE_BASE,
#     tokenizer_name=EMBEDDING_MODEL_NAME,
# )

# # Visualize the chunk sizes
# tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
# lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
# plt.hist(lengths, bins=30)  # You can adjust the number of bins as needed
# plt.title("Distribution of document lengths in the knowledge base (in count of tokens)")
# plt.xlabel("Token Count")
# plt.ylabel("Frequency")
# plt.show()

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
# import numpy as np
# import plotly.express as px

# def main():
#     pd.set_option("display.max_colwidth", None)  # This will be helpful when visualizing retriever outputs
#     ds = datasets.load_dataset("summydev/lecturersdata", split="train")
#     print(ds.column_names)  # This will show the names of the columns in the dataset
#     print(ds[0])  # Print the first record to see its structure

#     RAW_KNOWLEDGE_BASE = [
#         LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
#     ]

#     EMBEDDING_MODEL_NAME = "thenlper/gte-small"
#     MARKDOWN_SEPARATORS = [
#         "\n#{1,6} ",
#         "```\n",
#         "\n\\*\\*\\*+\n",
#         "\n---+\n",
#         "\n___+\n",
#         "\n\n",
#         "\n",
#         " ",
#         "",
#     ]

#     def split_documents(
#         chunk_size: int,
#         knowledge_base: List[LangchainDocument],
#         tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
#     ) -> List[LangchainDocument]:
#         text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
#             AutoTokenizer.from_pretrained(tokenizer_name),
#             chunk_size=chunk_size,
#             chunk_overlap=int(chunk_size / 10),
#             add_start_index=True,
#             strip_whitespace=True,
#             separators=MARKDOWN_SEPARATORS,
#         )
        
#         docs_processed = []
#         for doc in knowledge_base:
#             docs_processed += text_splitter.split_documents([doc])

#         # Remove duplicates
#         unique_texts = {}
#         docs_processed_unique = []
#         for doc in docs_processed:
#             if doc.page_content not in unique_texts:
#                 unique_texts[doc.page_content] = True
#                 docs_processed_unique.append(doc)

#         return docs_processed_unique

#     docs_processed = split_documents(
#         512,  # We choose a chunk size adapted to our model
#         RAW_KNOWLEDGE_BASE,
#         tokenizer_name=EMBEDDING_MODEL_NAME,
#     )

#     tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
#     lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
#     fig = pd.Series(lengths).hist()

#     embedding_model = HuggingFaceEmbeddings(
#             model_name=EMBEDDING_MODEL_NAME,
#             multi_process=True,
#             model_kwargs={"device": "cpu"},
#             encode_kwargs={"normalize_embeddings": True},
#         )

#     KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
#             docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
#         )

#     user_query = "which professor takes biology"
#     query_vector = embedding_model.embed_query(user_query)

#     print(f"\nStarting retrieval for {user_query=}...")
#     retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
#     print("\n==================================Top document==================================")
#     print(retrieved_docs[0].page_content)
#     print("==================================Metadata==================================")
#     print(retrieved_docs[0].metadata)

# if __name__ == '__main__':
#     main()

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
import numpy as np
import plotly.express as px

def main():
    pd.set_option("display.max_colwidth", None)  # This will be helpful when visualizing retriever outputs
    ds = datasets.load_dataset("summydev/lecturersdata", split="train")
    print(ds.column_names)  # This will show the names of the columns in the dataset
    print(ds[0])  # Print the first record to see its structure

    # Update the document creation to correctly access instructor names and course titles
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

    EMBEDDING_MODEL_NAME = "thenlper/gte-small"
    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    def split_documents(
        chunk_size: int,
        knowledge_base: List[LangchainDocument],
        tokenizer_name: Optional[str] = EMBEDDING_MODEL_NAME,
    ) -> List[LangchainDocument]:
        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            AutoTokenizer.from_pretrained(tokenizer_name),
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size / 10),
            add_start_index=True,
            strip_whitespace=True,
            separators=MARKDOWN_SEPARATORS,
        )
        
        docs_processed = []
        for doc in knowledge_base:
            docs_processed += text_splitter.split_documents([doc])

        # Remove duplicates
        unique_texts = {}
        docs_processed_unique = []
        for doc in docs_processed:
            if doc.page_content not in unique_texts:
                unique_texts[doc.page_content] = True
                docs_processed_unique.append(doc)

        return docs_processed_unique

    docs_processed = split_documents(
        512,  # We choose a chunk size adapted to our model
        RAW_KNOWLEDGE_BASE,
        tokenizer_name=EMBEDDING_MODEL_NAME,
    )

    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
    lengths = [len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)]
    fig = pd.Series(lengths).hist()

    embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            multi_process=True,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
            docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
        )

    user_query = "which professor takes health"
    query_vector = embedding_model.embed_query(user_query)

    print(f"\nStarting retrieval for {user_query=}...")
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    
    for i, doc in enumerate(retrieved_docs):
        print(f"\n==================================Top document {i+1}==================================")
        print(doc.page_content)
        print("==================================Instructor==================================")
        print(doc.metadata.get("instructor_name", "N/A"))  # Safely access the instructor's name
        print("==================================Course Title==================================")
        print(doc.metadata.get("course_title", "N/A"))  # Safely access the course title
        print("==================================Source==================================")
        print(doc.metadata.get("source", "N/A"))  # Safely access the source URL

if __name__ == '__main__':
    main()
 




# # Required Imports
# from transformers import  AutoTokenizer, AutoModelForCausalLM, pipeline
# from datasets import load_dataset
# from tqdm import tqdm
# from dask import delayed, compute
# import pandas as pd
# import numpy as np
# import torch
 
# from langchain.docstore.document import Document as LangchainDocument
# from langchain.text_splitter import RecursiveCharacterTextSplitter
 
# from langchain_community.vectorstores import FAISS

# from langchain_community.embeddings import HuggingFaceEmbeddings

# from langchain.vectorstores.faiss import DistanceStrategy
# from accelerate import BitsAndBytesConfig
# from ragatouille import RAGPretrainedModel

# pd.set_option("display.max_colwidth", None)

# # Variables
# RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
# READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
# EMBEDDING_MODEL_NAME = "thenlper/gte-small"
# MARKDOWN_SEPARATORS = [
#     "\n#{1,6} ",
#     "```\n",
#     "\n\\*\\*\\*+\n",
#     "\n---+\n",
#     "\n___+\n",
#     "\n\n",
#     "\n",
#     " ",
#     "",
# ]

# def split_documents(chunk_size: int, knowledge_base: list, tokenizer_name: str = EMBEDDING_MODEL_NAME):
#     tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
#     text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
#         tokenizer,
#         chunk_size=chunk_size,
#         chunk_overlap=int(chunk_size / 10),
#         add_start_index=True,
#         strip_whitespace=True,
#         separators=MARKDOWN_SEPARATORS,
#     )

#     # Use Dask for parallel processing
#     docs_processed = compute([delayed(text_splitter.split_documents)([doc]) for doc in tqdm(knowledge_base)])[0]
    
#     # Remove duplicates
#     unique_texts = {}
#     docs_processed_unique = []
#     for doc in docs_processed:
#         if doc.page_content not in unique_texts:
#             unique_texts[doc.page_content] = True
#             docs_processed_unique.append(doc)
    
#     return docs_processed_unique

# def main():
#     # Load the dataset
#     ds = load_dataset("m-ric/huggingface_doc", split="train")
    
#     print(ds.column_names)  # Show column names
#     print(ds[0])  # Print the first record to see its structure

#     # Process dataset into RAW_KNOWLEDGE_BASE
#     RAW_KNOWLEDGE_BASE = [
#         LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(ds)
#     ]
    
#     # Split documents
#     docs_processed = split_documents(512, RAW_KNOWLEDGE_BASE, tokenizer_name=EMBEDDING_MODEL_NAME)
    
#     # Use NumPy to optimize length calculations
#     tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
#     lengths = np.array([len(tokenizer.encode(doc.page_content)) for doc in tqdm(docs_processed)])
    
#     # Visualize with Pandas
#     fig = pd.Series(lengths).hist()

#     # Use CPU for embedding model
#     embedding_model = HuggingFaceEmbeddings(
#         model_name=EMBEDDING_MODEL_NAME,
#         multi_process=True,
#         model_kwargs={"device": "cpu"},  # Use CPU
#         encode_kwargs={"normalize_embeddings": True},
#     )

#     # Create FAISS vector store
#     KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
#         docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
#     )

#     # Embed a user query
#     user_query = "How to create a pipeline object?"
#     query_vector = embedding_model.embed_query(user_query)

#     print(f"\nStarting retrieval for {user_query=}...")
#     retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    
#     # Print results
#     print("\n==================================Top document==================================")
#     print(retrieved_docs[0].page_content)
#     print("==================================Metadata==================================")
#     print(retrieved_docs[0].metadata)

#     # Configure the LLM model for generating answers
#     bnb_config = BitsAndBytesConfig(
#         load_in_4bit=True,
#         bnb_4bit_use_double_quant=True,
#         bnb_4bit_quant_type="nf4",
#         bnb_4bit_compute_dtype=torch.bfloat16,
#     )
#     model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
#     tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

#     READER_LLM = pipeline(
#         model=model,
#         tokenizer=tokenizer,
#         task="text-generation",
#         do_sample=True,
#         temperature=0.2,
#         repetition_penalty=1.1,
#         return_full_text=False,
#         max_new_tokens=500,
#     )

#     # Run a test query
#     READER_LLM("What is 4+4? Answer:")

#     # Prompt structure for the RAG system
#     prompt_in_chat_format = [
#         {
#             "role": "system",
#             "content": """Using the information contained in the context,
#             give a comprehensive answer to the question.
#             Respond only to the question asked, response should be concise and relevant to the question.
#             Provide the number of the source document when relevant.
#             If the answer cannot be deduced from the context, do not give an answer.""",
#         },
#         {
#             "role": "user",
#             "content": """Context:
#             {context}
#             ---
#             Now here is the question you need to answer.
    
#             Question: {question}""",
#         },
#     ]
    
#     RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
#         prompt_in_chat_format, tokenize=False, add_generation_prompt=True
#     )
    
#     # Create final prompt
#     retrieved_docs_text = [doc.page_content for doc in retrieved_docs]  # We only need the text of the documents
#     context = "\nExtracted documents:\n"
#     context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

#     final_prompt = RAG_PROMPT_TEMPLATE.format(question="How to create a pipeline object?", context=context)

#     # Redact an answer
#     answer = READER_LLM(final_prompt)[0]["generated_text"]
#     print(answer)
#     print("################### Reranking #####################")
    
#     def answer_with_rag(
#         question: str,
#         llm: pipeline,
#         knowledge_index: FAISS,
#         reranker: RAGPretrainedModel = None,
#         num_retrieved_docs: int = 30,
#         num_docs_final: int = 5,
#     ):
#         # Gather documents with retriever
#         print("=> Retrieving documents...")
#         relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
#         relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

#         # Optionally rerank results
#         if reranker:
#             print("=> Reranking documents...")
#             relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
#             relevant_docs = [doc["content"] for doc in relevant_docs]

#         relevant_docs = relevant_docs[:num_docs_final]

#         # Build the final prompt
#         context = "\nExtracted documents:\n"
#         context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

#         final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

#         # Generate the answer
#         print("=> Generating answer...")
#         answer = llm(final_prompt)[0]["generated_text"]

#         return answer, relevant_docs

#     # Run the RAG process with reranking
#     question = "How to create a pipeline object?"
#     answer, relevant_docs = answer_with_rag(question, READER_LLM, KNOWLEDGE_VECTOR_DATABASE, reranker=RERANKER)
#     print(answer)

# if __name__ == "__main__":
#     main()
