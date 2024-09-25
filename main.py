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

#     # Update the document creation to correctly access instructor names and course titles
#     RAW_KNOWLEDGE_BASE = [
#         LangchainDocument(
#             page_content=doc["text"],
#             metadata={
#                 "source": doc["source"],
#                 "instructor": doc["__index_level_1__"],  # Access instructor name
#                 "course_title": doc["__index_level_0__"].split(",")[0]  # Extract course title from the first part
#             }
#         ) for doc in tqdm(ds)
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

#     user_query = "which professor takes statistics"
#     query_vector = embedding_model.embed_query(user_query)

#     print(f"\nStarting retrieval for {user_query=}...")
#     retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    
#     for i, doc in enumerate(retrieved_docs):
#         print(f"\n==================================Top document {i+1}==================================")
#         print(doc.page_content)
#         print("==================================Instructor==================================")
#         print(doc.metadata.get("instructor", "N/A"))  # Safely access the instructor's name
#         print("==================================Course Title==================================")
#         print(doc.metadata.get("course_title", "N/A"))  # Safely access the course title
#         print("==================================Source==================================")
#         print(doc.metadata.get("source", "N/A"))  # Safely access the source URL

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
from transformers import pipeline
import torch
from transformers import AutoModelForCausalLM

def main():
    pd.set_option("display.max_colwidth", None)  # This will be helpful when visualizing retriever outputs
    ds = datasets.load_dataset("summydev/lecturersdata", split="train")
    print(ds.column_names)  # This will show the names of the columns in the dataset
    print(ds[0])  # Print the first record to see its structure

    RAW_KNOWLEDGE_BASE = [
        LangchainDocument(
            page_content=doc["description"],  # Using description as content
            metadata={
                "source": doc["source"],
                "instructor_name": doc["instructorname"],  # Updated key
                "course_title": doc["coursetitle"],        # Updated key
                "rating": doc["rating"],                    # Updated key
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

    user_query = "which professor takes statistics"
    query_vector = embedding_model.embed_query(user_query)

    print(f"\nStarting retrieval for {user_query=}...")
    retrieved_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(query=user_query, k=5)
    print("\n==================================Top document==================================")
    print(retrieved_docs[0].page_content)
    print("==================================Metadata==================================")
    print(retrieved_docs[0].metadata)

    READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

    # Load model without quantization
    model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)

    READER_LLM = pipeline(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        do_sample=True,
        temperature=0.2,
        repetition_penalty=1.1,
        return_full_text=False,
        max_new_tokens=500,
    )

    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
provide a concise and relevant answer to the question.
Mention the instructor's name, description, rating, and course title if applicable.
If the answer cannot be directly deduced from the context, provide the most relevant related information instead.""",
        },
        {
            "role": "user",
            "content": """Context:
{context}
---
Now here is the question you need to answer.

Question: {question}""",
        },
    ]

    retrieved_docs_text = [doc.page_content for doc in retrieved_docs]  # We only need the text of the documents
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(retrieved_docs_text)])

    final_prompt = prompt_in_chat_format[1]["content"].format(question="which professor takes statistics", context=context)

    # Redact an answer
    answer = READER_LLM(final_prompt)[0]["generated_text"]
    print(answer)

if __name__ == '__main__':
    main()
