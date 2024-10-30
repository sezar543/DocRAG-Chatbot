print("Test: FastAPI app is starting...")

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi import FastAPI, File, Request, Header, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import json
from pydantic import BaseModel
from typing import List, Any, Optional
import rag_util
from model import ChatModel
from rag_util import Encoder, PGVectorDb
from fastapi.encoders import jsonable_encoder
import atexit
import shutil
from fastapi_cache import FastAPICache  # caching
from fastapi_cache.decorator import cache
from fastapi_cache.backends.inmemory import InMemoryBackend
import logging
import threading

# Add the directory containing `rag_util.py` to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Environment settings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# import logging
# logging.basicConfig(level=logging.INFO)

# from transformers import logging as transformers_logging
# transformers_logging.set_verbosity_info()

# Add the directory containing `rag_util.py` to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from custom_json_encoder import CustomJsonEncoder

# fastapi_host = os.getenv("FASTAPI_HOST", "localhost")

app = FastAPI()

# In-memory storage for session-specific DB instances
session_dbs = {}
session_locks = {}
session_flags = {}

# Ensure thread safety
session_lock = threading.Lock()

# Custom cleanup function
def cleanup_files():
    if os.path.exists(FILES_DIR):
        shutil.rmtree(FILES_DIR)
    with session_lock:
        for session_id, db in session_dbs.items():
            db.delete_collection(session_id)
        session_dbs.clear()

# atexit.register(cleanup_files)

# Define a custom JSON response class that uses the CustomJsonEncoder
class CustomJSONResponse(JSONResponse):
    def render(self, content: Any) -> bytes:
        return json.dumps(jsonable_encoder(content), cls=CustomJsonEncoder).encode("utf-8")


@app.on_event("startup")
async def startup():
    FastAPICache.init(InMemoryBackend(), prefix="fastapi-cache")
    global model, encoder
    # model = await load_model()
    # encoder = await load_encoder()
    model = ChatModel.from_dict(await load_model())
    encoder = Encoder.from_dict(await load_encoder())
    print("Models loaded successfully.")

# Setup file directory
FILES_DIR = "./uploaded_files"
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
    
print("CACHE_DIR = ", CACHE_DIR)

# Ensure the directory exists
os.makedirs(FILES_DIR, exist_ok=True)

POSTGRES_USER = os.getenv('POSTGRES_USER')
POSTGRES_PASSWORD = os.getenv('POSTGRES_PASSWORD')
POSTGRES_DB = os.getenv('POSTGRES_DB')
POSTGRES_HOST = os.getenv('DB_HOST')
POSTGRES_PORT = os.getenv('DB_PORT')

model_encoder = os.getenv("model_encoder")
LLM_model_id = os.getenv("LLM_model_id")
if not LLM_model_id:
    raise ValueError("The LLM_MODEL_ID environment variable is not set.")

print(f"Loading model with ID: {LLM_model_id}")

if not all([POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_DB, POSTGRES_HOST, POSTGRES_PORT]):
    raise ValueError("Database configuration is incomplete. Please set the required environment variables.")

# def cleanup_files():
#     if os.path.exists(FILES_DIR):
#         shutil.rmtree(FILES_DIR)
#     # Add any other cleanup code for your database/context here

# atexit.register(cleanup_files)


print("Loading models...")

if torch.cuda.is_available():
    print("device is cuda!!!!!!!!!!!!!!")
    device = "cuda"
else:
    raise RuntimeError("No GPU found. A GPU is needed for quantization.")

# @cache()
# async def load_model() -> Any:
#     return ChatModel(LLM_model_id=LLM_model_id, device="cuda")

# @cache()
# async def load_encoder() -> Any:
#     return rag_util.Encoder(model_name=model_encoder, device="cpu")
@cache()
async def load_model() -> dict:
    model_instance = ChatModel(LLM_model_id=LLM_model_id, device="cuda")
    return model_instance.to_dict()

@cache()
async def load_encoder() -> dict:
    encoder_instance = rag_util.Encoder(model_name=model_encoder, device="cpu")
    return encoder_instance.to_dict()

ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print("Models loaded successfully.")
    

async def get_session_id(session_id: str = Header(None)):
    if session_id is None:
        raise HTTPException(status_code=400, detail="Session ID header missing")
    return session_id

           
class ResetConversationRequest(BaseModel):
    session_id: str
    k: int

class QueryRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    k: int = 3
    temperature: float = 0.01

# class CachedResponse(BaseModel):
#     response: str

@app.get("/")
async def read_root():
    return {"Hello": "World!!!!!!!!!"}

# @app.post("/uploadfile/")
# async def upload_file(request: Request, uploaded_file: UploadFile = File(...), x_session_id: str = Header(None)):
#     print("FastAPI: session_id in method upload_file is ", x_session_id)
#     headers = request.headers
#     user_dir = os.path.join(FILES_DIR, x_session_id)
#     print("FastAPI: user_dir is ", user_dir)
#     if not os.path.exists(user_dir):
#         os.makedirs(user_dir)

#     file_path = os.path.join(user_dir, uploaded_file.filename)
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(uploaded_file.file, buffer)

#     return {"name": uploaded_file.filename, "path": file_path}

def cleanup_session(session_id):
    user_dir = os.path.join(FILES_DIR, session_id)
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
    with session_lock:
        if session_id in session_dbs:
            session_dbs[session_id].delete_collection(session_id)
            del session_dbs[session_id]
            del session_locks[session_id]
            del session_flags[session_id]

def create_session(session_id):
    if session_id not in session_locks:
        session_locks[session_id] = threading.Lock()
        session_flags[session_id] = 0
        logging.debug(f"Session ID {session_id} initialized and added to session_locks")

@app.post("/uploadfile/")
async def upload_file(uploaded_file: UploadFile = File(...), x_session_id: str = Header(None)):
    user_dir = os.path.join(FILES_DIR, x_session_id)
    if not os.path.exists(user_dir):
        os.makedirs(user_dir)

    file_path = os.path.join(user_dir, uploaded_file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)

    # docs = rag_util.load_and_split_pdfs([file_path])  # Corrected to pass a list of file paths
    # embedding_function = encoder.embedding_function
    with session_lock:
        session_flags[x_session_id] = 0
        if x_session_id not in session_dbs:
            # session_dbs[x_session_id] = PGVectorDb(docs=docs, embedding_function=embedding_function, session_id=x_session_id)
            session_locks[x_session_id] = threading.Lock()
        # else:
        #     session_dbs[x_session_id] = PGVectorDb(docs=docs, embedding_function=embedding_function, session_id=x_session_id)

    return {"name": uploaded_file.filename, "path": file_path}
    
# @app.delete("/deletefile/{file_name}")
# async def delete_file(request: Request, file_name: str, x_session_id: str = Header(None)):
#     print("FastAPI: session_id in method delete_file is ", x_session_id)
#     headers = request.headers
#     user_dir = os.path.join(FILES_DIR, x_session_id)
#     file_path = os.path.join(user_dir, file_name)
#     print("FastAPI: file_path is ", file_path)
#     if os.path.exists(file_path):
#         os.remove(file_path)
#         print(f"File {file_path} deleted successfully.")  # Log successful deletion
#         return {"detail": f"File {file_name} deleted"}
#     else:
#         print(f"File {file_path} not found.")  # Log file not found
#         raise HTTPException(status_code=404, detail=f"File {file_name} not found")

@app.delete("/deletefile/{file_name}")
async def delete_file(file_name: str, x_session_id: str = Header(None)):
    user_dir = os.path.join(FILES_DIR, x_session_id)
    file_path = os.path.join(user_dir, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        with session_lock:
            if x_session_id in session_dbs:
                with session_locks[x_session_id]:
                    session_dbs[x_session_id].delete_collection(x_session_id)
                    del session_dbs[x_session_id]
                    del session_locks[x_session_id]
                    del session_flags[x_session_id]
                    session_flags[x_session_id] = 0  # Reset flag to 0 after deleting DB
        return {"detail": f"File {file_name} deleted"}
    else:
        raise HTTPException(status_code=404, detail=f"File {file_name} not found")


# @app.on_event("shutdown")
# async def shutdown_event():
#     # Cleanup all sessions on shutdown
#     with session_lock:
#         for session_id in list(session_dbs.keys()):
#             cleanup_session(session_id)

# Ensure atexit cleanup in case shutdown_event does not trigger
# atexit.register(shutdown_event)
# Combine cleanup functions
def combined_cleanup():
    cleanup_files()
    shutdown_event()

atexit.register(combined_cleanup)

@app.post("/reset_conversation")
async def reset_conversation(request: dict):
    try:
        data = await request.json()
        # user_id = data.get("user_id")
        k = data.get("k", 3)  # Get 'k' from request, default to 3 if not provided
        print("k inside reset conversation of fastapi is ", k)
        session_id = request["session_id"]
        if session_id is None:
            return {"error": "session_id is required"}

        chat_model.reset_memory(user_id=session_id, memory_reset_trigger=True, k=k)

        chat_model.reset_memory(user_id=session_id, memory_reset_trigger=True, k=k)
        
        return {"message": "Conversation memory reset and session-specific database cleaned up"}

    except Exception as e:
        return {"error": str(e)}

    # session_id = request["session_id"]
    # k = request["k"]
    # print("session_id inside reset_conversation of fastapi=", session_id)
    # memory_reset_trigger = True
    # model.check_memory(user_id=session_id, memory_reset_trigger=True, k=k)
    # return {"message": "Conversation memory reset"}

def load_and_split_pdfs(session_id):
    user_dir = os.path.join(FILES_DIR, session_id)
    file_paths = [os.path.join(user_dir, f) for f in os.listdir(user_dir) if f.endswith(".pdf")]
    if file_paths:
        return rag_util.load_and_split_pdfs(file_paths)
    else:
        return []

# @app.post("/query")
# Update the /query endpoint to use CustomJSONResponse
@app.post("/query", response_class=CustomJSONResponse)
async def query(request: Request, query: QueryRequest, x_session_id: str = Header(None)):
    # if not x_session_id:
    #     raise HTTPException(status_code=400, detail="Session ID not found")

    create_session(x_session_id)  # Ensure session is created if not exists

    headers = request.headers

    print("query.k = ", query.k)
    logging.info(f"Headers received: {headers}")
    user_prompt = query.prompt
    max_new_tokens = query.max_new_tokens
    k = query.k
    temperature = query.temperature

    print("user_prompt = ", user_prompt)

    logging.info(f"Received request data: {request}")
    response_data = {"response": "Hello, World!"}  # Replace with your actual logic
    logging.info(f"Sending response data: {response_data}")

    context = None
    database = None
    # Check for any files in the directory
    file_paths = []

    session_path = os.path.join(FILES_DIR, x_session_id)
    if os.path.isdir(session_path):
        print("session_path = ", session_path)
        for file in os.listdir(session_path):
            if file.endswith(".pdf"):
                file_paths.append(os.path.join(session_path, file))

    with session_locks[x_session_id]:
        print("with session_locks[x_session_id]!")
        print("length file_paths = ", len(file_paths))
        print("session_flags[x_session_id] = ", session_flags[x_session_id])
        if session_flags[x_session_id] == 0 and file_paths:
            try:
                docs = rag_util.load_and_split_pdfs(file_paths)
                print("docs = ", len(docs))
                embedding_function = encoder.embedding_function
                session_dbs[x_session_id] = PGVectorDb(docs=docs, embedding_function=embedding_function, session_id=x_session_id)
                session_flags[x_session_id] = 1  # Set flag to 1 after creating DB
            except ValueError as e:
                logging.error(f"Error processing PDFs: {e}")
                raise HTTPException(status_code=400, detail=str(e))
            except Exception as e:
                logging.error(f"Unexpected error: {e}")
                raise HTTPException(status_code=500, detail="Internal Server Error")
        
            database = session_dbs[x_session_id].get_db()
            context = database.similarity_search(user_prompt, k=query.k)

        answer = await model.generate(
            user_id=x_session_id,
            question=user_prompt,
            db=database,
            context=context,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
    return {"response": answer} 


# Error handling
@app.exception_handler(Exception)
async def exception_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"message": str(exc)})

# @app.post("/query", response_model=CachedResponse)
# async def query_model(query: QueryRequest):
#     user_prompt = query.prompt
#     print("user_prompt = ", user_prompt)

#     logging.info(f"Received request data: {query}")
#     response_data = {"response": "Hello, World!"}  # Replace with your actual logic
#     logginfg.info(f"Sending response data: {response_data}")

#     context = None
#     database = None

#     file_paths = [os.path.join(FILES_DIR, f) for f in os.listdir(FILES_DIR) if f.endswith(".pdf")]
#     if file_paths:
#         print("Found PDF files:", file_paths)
#         docs = rag_util.load_and_split_pdfs(file_paths)
#         encoder_instance = encoder
#         embedding_function = encoder_instance.embedding_function
#         DB = rag_util.PGVectorDb(docs=docs, embedding_function=embedding_function)
#         context = DB.similarity_search(user_prompt, k=query.k)
#         database = DB.get_db()
#     else:
#         print("No PDF files found in directory:", FILES_DIR)

#     # Ensure actual_model is correctly set
#     if not model:
#         raise HTTPException(status_code=500, detail="Model not loaded correctly")
        
#     # actual_model = model

#     answer = await model.generate(
#         user_prompt,
#         db=database,
#         context=context,
#         max_new_tokens=query.max_new_tokens,
#         temperature=query.temperature,
#         k=query.k,
#     )

#     return {"response": answer}

# @app.post("/query", response_model=CachedResponse)
# @cache(expire=60)  # Cache responses for 60 seconds
# async def query_model(request_data: QueryRequest):
#     user_prompt = request_data.prompt
#     print("user_prompt = ", user_prompt)

#     logging.info(f"Received request data: {request_data}")
#     response_data = {"response": "Hello, World!"}  # Replace with your actual logic
#     logging.info(f"Sending response data: {response_data}")

#     context = None
#     database = None

#     # Check for any files in the directory
#     file_paths = [os.path.join(FILES_DIR, f) for f in os.listdir(FILES_DIR) if f.endswith(".pdf")]
#     if file_paths:
#         print("Found PDF files:", file_paths)
#         docs = rag_util.load_and_split_pdfs(file_paths)
#         # encoder_instance = await encoder  # If encoder is a coroutine
#         encoder_instance = encoder  # If encoder is a coroutine

#         embedding_function = encoder_instance.embedding_function  # No need to await
#         DB = rag_util.PGVectorDb(docs=docs, embedding_function=embedding_function)
#         context = DB.similarity_search(user_prompt, k=request_data.k)
#         database = DB.get_db()
#     else:
#         print("No PDF files found in directory:", FILES_DIR)

#     # actual_model = await model
#     actual_model = model

#     answer = await actual_model.generate(
#         user_prompt,
#         db=database,
#         context=context,
#         max_new_tokens=request_data.max_new_tokens,
#         temperature=request_data.temperature,
#         k=request_data.k,
#     )











# @app.post("/query")
# def query_model(request: QueryRequest):
#     # Ensure the directory exists before listing its contents
#     os.makedirs(FILES_DIR, exist_ok=True)

#     user_prompt = request.prompt
#     print("user_prompt =", user_prompt)

#     logging.info(f"Received request data: {request}")

#     context = None
#     database = None
#     context_docs = None

#     # Check for any files in the directory
#     # file_paths = [os.path.join(FILES_DIR, f) for f in os.listdir(FILES_DIR) if f.endswith(".pdf")]

#     # Check for any files in the directory
#     file_paths = []
#     for session_dir in os.listdir(FILES_DIR):
#         session_path = os.path.join(FILES_DIR, session_dir)
#         if os.path.isdir(session_path):
#             for file in os.listdir(session_path):
#                 if file.endswith(".pdf"):
#                     file_paths.append(os.path.join(session_path, file))
#     if not file_paths:
#         raise HTTPException(status_code=400, detail="No PDF files found.")

#     # file_paths = [os.path.join(FILES_DIR, f) for f in os.listdir(FILES_DIR) if f.endswith(".pdf") and session_id in f]
  
#     print("file_paths = ", file_paths)

#     if file_paths:
#         print("Found PDF files:", file_paths)
#         docs = rag_util.load_and_split_pdfs(file_paths)
#         print("We got here! Just before creating DB!")

#         try:
#             DB = rag_util.PGVectorDb(docs=docs, embedding_function=encoder.embedding_function)
#             context_docs = DB.similarity_search(user_prompt, k=request.k)
#             database = DB.get_db()
#         except Exception as e:
#             print(f"Error creating database: {e}")

#         # Use context_docs as needed
#     else:
#         print("No PDF files found in directory:", FILES_DIR)

    
#     # if file_paths:
#     #     print("Found PDF files:", file_paths)
#     #     docs = rag_util.load_and_split_pdfs(file_paths)
#     #     print("We got here! Just before creating DB!")
#     #     DB = rag_util.PGVectorDb(docs=docs, embedding_function=encoder.embedding_function)
#     #     print("I am not sure whether we got here! Just after creating DB!")
#     #     context_docs = DB.similarity_search(user_prompt, k=request.k)
#     #     # context = " ".join([doc.page_content for doc in context_docs if hasattr(doc, 'page_content')])
        
#     #     database = DB.get_db()
#     # else:
#     #     print("No PDF files found in directory:", FILES_DIR)

#     # Construct the payload for the Hugging Face API
#     if context_docs != None or context_docs != "":
#         print("context_docs is ", context_docs)

# #         contextualize_q_system_prompt = """Given a chat history and the latest user question \
# # which might reference context in the chat history, formulate a standalone question \
# # which can be understood without the chat history. Do NOT answer the question, \
# # just reformulate it if needed and otherwise return it as is."""
# #         contextualize_q_prompt = ChatPromptTemplate.from_messages(
# #             [
# #                 ("system", contextualize_q_system_prompt),
# #                 MessagesPlaceholder("chat_history"),
# #                 ("human", "{input}"),
# #             ]
# #         )
# #         history_aware_retriever = create_history_aware_retriever(
# #             self.llm, db.as_retriever(search_kwargs={"k": k,"search_type": "similarity"}), contextualize_q_prompt
# #         )

#         qa_system_prompt = """You are an assistant for question-answering tasks. \
# Use the following pieces of retrieved context to answer the question. \
# If you don't know the answer, just say that you don't know. \
# Use three sentences maximum and keep the answer concise.\

# {context}"""

#         qa_prompt = qa_system_prompt.format(context=context_docs)
#         payload = {
#             "inputs": f"{qa_prompt}\nHuman: {user_prompt}\nAI:",
#             "parameters": {
#                 "max_new_tokens": request.max_new_tokens,
#                 "temperature": request.temperature,
#             },
#             "options": {"wait_for_model": True}
#         }

#     else:
#         general_prompt_template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.  

# Current conversation:
# {history}
# Human: {input}
# AI:"""
#         general_prompt = general_prompt_template.format(history="", input=user_prompt)
#         payload = {
#             "inputs": general_prompt,
#             "parameters": {
#                 "max_new_tokens": request.max_new_tokens,
#                 "temperature": request.temperature,
#             },
#             "options": {"wait_for_model": True}
#         }
#     headers = {
#         "Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"
#     }

#     # # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
#     # model_id = "mistralai/Mistral-7B-Instruct-v0.2"
#     # API_URL = f"https://api-inference.huggingface.co/models/{model_id}"

#     # response = requests.post(API_URL, headers=headers, json=payload)
#     # print("Hooooooy! response is: ", response)

#     # question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

#     # rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#     # store = {}

#     # def get_session_history(session_id: str) -> BaseChatMessageHistory:
#     #     if session_id not in store:
#     #         store[session_id] = ChatMessageHistory()
#     #     return store[session_id]

#     # conversational_rag_chain = RunnableWithMessageHistory(
#     #     rag_chain,
#     #     get_session_history,
#     #     input_messages_key="input",
#     #     history_messages_key="chat_history",
#     #     output_messages_key="answer",
#     # )
#     # response = conversational_rag_chain.invoke(
#     #     {"input": question},
#     #     config={
#     #         "configurable": {"session_id": "abc123"}
#     #     },  # constructs a key "abc123" in `store`.
#     # )
#     # response = response["answer"].split('Assistant: ')[-1].strip()


#     try:
#         data = response.json()
#         if 'error' in data:
#             return {"response": f"Error: {data['error']}"}

#         generated_text = data[0]["generated_text"]
#         if context != None or context != "":
#             print("Attention!!!!!!!!!!!! You got here!")
#             # answer = generated_text[len(context_docs + user_prompt):].strip()
#             ai_response_start = generated_text.find("AI:") + len("AI:")
#             answer = generated_text[ai_response_start:].strip()

#             # # Extract the last paragraph from the response
#             # last_paragraph = generated_text.split('\n')[-1].strip()
#             # answer = last_paragraph
#         else:
#             print("Attention!!!!!!!!!!!! You DID NOT get there!")
#             answer = generated_text.strip()
#     except Exception as e:
#         answer = f"Error processing response: {str(e)}"
#         print("Error response content:", response.text)

#     return {"response": answer}







# @app.post("/uploadfile/")
# async def upload_file(uploaded_file: UploadFile = File(...)):
#     file_path = os.path.join(FILES_DIR, uploaded_file.filename)
#     os.makedirs(FILES_DIR, exist_ok=True)
#     with open(file_path, "wb") as f:
#         f.write(await uploaded_file.read())
#     print("File uploaded successfully:", uploaded_file.filename)

#     return {"filename": uploaded_file.filename}




