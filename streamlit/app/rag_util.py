import os
import glob
import re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from transformers import AutoTokenizer
from dotenv import load_dotenv
from pathlib import Path
from sqlalchemy.exc import OperationalError

# import logging
# logging.basicConfig(level=logging.INFO)

# from transformers import logging as transformers_logging
# transformers_logging.set_verbosity_info()

load_dotenv()

# CACHE_DIR = os.path.normpath(
#     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
# )

# Use os.getenv to retrieve TRANSFORMERS_CACHE environment variable
CACHE_DIR = os.getenv("TRANSFORMERS_CACHE", None)

# If TRANSFORMERS_CACHE is not set, fallback to the default directory
if CACHE_DIR is None:
    CACHE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    )

# model_name = "sentence-transformers/all-MiniLM-L12-v2"
# model_encoder = "sentence-transformers/all-mpnet-base-v2"
model_name = os.getenv("model_encoder")


class Encoder:
    def __init__(
        self, model_name: str = model_name, device="cpu"
    ):
        self.model_name = model_name
        self.device = device
        # self.check_cache()

        self.embedding_function = HuggingFaceEmbeddings(
            model_name=model_name,
            cache_folder=CACHE_DIR,
            model_kwargs={"device": device},
        )

    def to_dict(self):
        return {
            "model_name": self.model_name,
            "device": self.device,
            # Add other necessary attributes here
        }
        
    def from_dict(data):
        # Example implementation, adjust according to your needs
        model_name = data.get('model_name', 'sentence-transformers/all-MiniLM-L12-v2')
        device = data.get('device', 'cpu')
        return Encoder(model_name=model_name, device=device)
        
    # def check_cache(self):
    #     # Path to cached model files
    #     cache_model_dir = os.path.join(CACHE_DIR, self.model_name.replace('/', '_'))
    #     cached_files = glob.glob(os.path.join(cache_model_dir, "*"))
        
    #     if cached_files:
    #         print(f"Model files found in cache: {cached_files}")
    #     else:
    #         print("Model files not found in cache. Model will be downloaded.")        

class PGVectorDb:
    def __init__(self, docs, embedding_function, session_id):
        postgres_user = os.getenv("POSTGRES_USER")
        postgres_database = os.getenv("POSTGRES_DB")
        postgres_pass = os.getenv("POSTGRES_PASSWORD")
        postgres_host = os.getenv("DB_HOST")
        postgres_port = os.getenv("DB_PORT", 5432)

        # POSTGRES_USER = os.getenv("POSTGRES_USER")
        # POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
        # POSTGRES_DB = os.getenv("POSTGRES_DB")
        # POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres")
        # POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")

        print(f"POSTGRES_USER: {postgres_user}")
        print(f"POSTGRES_PASS: {postgres_pass}")
        print(f"POSTGRES_DB: {postgres_database}")
        print(f"POSTGRES_HOST: {postgres_host}")
        print(f"POSTGRES_PORT: {postgres_port}")

        if not postgres_pass or not postgres_host or not postgres_port:
            raise ValueError("Database configuration is incomplete. Please set the required environment variables.")

        COLLECTION_NAME = f'pdf_collection_vectors_{session_id}'

        try:
            # First attempt with port included in connection string
            print("First attempt! CONNECTION_STRING is ")
            CONNECTION_STRING = f"postgresql+psycopg2://{postgres_user}:{postgres_pass}@{postgres_host}:{postgres_port}/{postgres_database}"
            print(CONNECTION_STRING)
            postgres_host = 'localhost:5432'
            CONNECTION_STRING = f"postgresql+psycopg2://{postgres_user}:{postgres_pass}@{postgres_host}/{postgres_database}"
            print("CONNECTION_STRING2 = ", CONNECTION_STRING )
            self.db = PGVector.from_documents(
                embedding=embedding_function, 
                documents=docs, 
                collection_name=COLLECTION_NAME,
                connection_string=CONNECTION_STRING,
                use_jsonb=True,
                pre_delete_collection=True,
            )
        except OperationalError as e1:
            print(f"First attempt failed: {e1}")
            time.sleep(5)  # Wait for 5 seconds before the second attempt
            try:
                # Fallback attempt without port
                print("Fallback attempt without port!")
                CONNECTION_STRING = f"postgresql+psycopg2://{postgres_user}:{postgres_pass}@{postgres_host}/{postgres_database}"
                print(CONNECTION_STRING)
                self.db = PGVector.from_documents(
                    embedding=embedding_function, 
                    documents=docs, 
                    collection_name=COLLECTION_NAME,
                    connection_string=CONNECTION_STRING,
                    use_jsonb=True,
                    pre_delete_collection=True,
                )
            except OperationalError as e2:
                print(f"Fallback attempt failed: {e2}")
                raise Exception(f"Both attempts to connect to the database failed. Error: {e2}") from e2
                
        self.db = PGVector.from_documents(embedding=embedding_function, 
            documents=docs, 
            collection_name=COLLECTION_NAME,
            connection_string=CONNECTION_STRING,
            use_jsonb=True,
            pre_delete_collection=True,
        )
        

    def similarity_search(self, question: str, k: int = 3):
        retrieved_docs = self.db.similarity_search(question, k=k)
        context = "".join(doc.page_content + "\n" for doc in retrieved_docs)
        return context
    
    def get_db(self):
        return self.db
    
    def delete_collection(self, session_id):
        COLLECTION_NAME = f'pdf_collection_vectors_{session_id}'
        try:
            self.db.delete_collection()
            print(f"Collection {COLLECTION_NAME} deleted successfully.")
        except Exception as e:
            print(f"Error deleting collection {COLLECTION_NAME}: {e}")


# # Mock implementation of PGVector for completeness
# class PGVector:
#     def __init__(self):
#         self.collections = {}

#     def from_documents(self, embedding, documents, collection_name, connection_string, use_jsonb, pre_delete_collection):
#         if pre_delete_collection and collection_name in self.collections:
#             del self.collections[collection_name]
#         self.collections[collection_name] = documents
#         return self

#     def similarity_search(self, question, k):
#         # Implement similarity search logic
#         pass

#     def drop_collection(self, collection_name):
#         if collection_name in self.collections:
#             del self.collections[collection_name]
#         else:
#             raise ValueError(f"Collection {collection_name} does not exist.")

def load_and_split_pdfs(file_paths: list, chunk_size: int = 1024):
    for file_path in file_paths:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"The file {file_path} is empty and cannot be processed.")        
        loaders = [PyPDFLoader(file_path) for file_path in file_paths]
    # loaders = [PyPDFLoader(file_path) for file_path in file_paths]
    pages = []
    for loader in loaders:
        pages.extend(loader.load())
    def normalize_spaces(doc):
        # Replace multiple occurrences of '\xa0' or spaces with a single space
        normalized_page_content = re.sub(r'\s+', ' ', doc.page_content.replace('‐‐', ' ').replace('--', ' '))
        return Document(metadata=doc.metadata,page_content=normalized_page_content)
    
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L12-v2"
        ),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        separators=['\n\n', '\n', r'(?=>\. )', '\xa0', ' ', ''],
        strip_whitespace=True,
    )
    normalized_pages = [normalize_spaces(page) for page in pages]
    docs = text_splitter.split_documents(normalized_pages)
    return docs
