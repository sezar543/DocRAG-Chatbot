print("Test: FastAPI app is starting...")

import os
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.chains import LLMChain, ConversationChain, create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from dotenv import load_dotenv
import re
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from transformers import AutoTokenizer

# Add the directory containing `rag_util.py` to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import rag_util
from pydantic import BaseModel
from typing import List
import requests

import rag_util
from model import ChatModel

import atexit
import shutil


from transformers import BertLMHeadModel, BertTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM

fastapi_host = os.getenv("FASTAPI_HOST", "localhost")

app = FastAPI()
# Initialize models

# Setup file directory
FILES_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files"))

# Ensure the directory exists
os.makedirs(FILES_DIR, exist_ok=True)

def cleanup_files():
    if os.path.exists(FILES_DIR):
        shutil.rmtree(FILES_DIR)
    # Add any other cleanup code for your database/context here

atexit.register(cleanup_files)


print("Loading models...")
# model = ChatModel(model_id="mistralai/Mistral-7B-Instruct-v0.2", device="cuda")
encoder = rag_util.Encoder(model_name="sentence-transformers/all-MiniLM-L12-v2", device="cpu")

ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

print("Models loaded successfully.")

class QueryRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 512
    k: int = 3
    temperature: float = 0.01

@app.get("/")
async def read_root():
    return {"Hello": "World!!!"}


@app.post("/query")
def query_model(request: QueryRequest):
    # Ensure the directory exists before listing its contents
    os.makedirs(FILES_DIR, exist_ok=True)

    user_prompt = request.prompt
    print("user_prompt =", user_prompt)

    logging.info(f"Received request data: {request}")

    context = None
    database = None

    # Check for any files in the directory
    file_paths = [os.path.join(FILES_DIR, f) for f in os.listdir(FILES_DIR) if f.endswith(".pdf")]

    context_docs = None

    if file_paths:
        print("Found PDF files:", file_paths)
        docs = rag_util.load_and_split_pdfs(file_paths)
        DB = rag_util.PGVectorDb(docs=docs, embedding_function=encoder.embedding_function)
        context_docs = DB.similarity_search(user_prompt, k=request.k)
        # context = " ".join([doc.page_content for doc in context_docs if hasattr(doc, 'page_content')])
        
        database = DB.get_db()
    else:
        print("No PDF files found in directory:", FILES_DIR)

    # Construct the payload for the Hugging Face API
    if context_docs != None or context_docs != "":
        print("context_docs is ", context_docs)

#         contextualize_q_system_prompt = """Given a chat history and the latest user question \
# which might reference context in the chat history, formulate a standalone question \
# which can be understood without the chat history. Do NOT answer the question, \
# just reformulate it if needed and otherwise return it as is."""
#         contextualize_q_prompt = ChatPromptTemplate.from_messages(
#             [
#                 ("system", contextualize_q_system_prompt),
#                 MessagesPlaceholder("chat_history"),
#                 ("human", "{input}"),
#             ]
#         )
#         history_aware_retriever = create_history_aware_retriever(
#             self.llm, db.as_retriever(search_kwargs={"k": k,"search_type": "similarity"}), contextualize_q_prompt
#         )

        qa_system_prompt = """You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\

{context}"""

        qa_prompt = qa_system_prompt.format(context=context_docs)
        payload = {
            "inputs": f"{qa_prompt}\nHuman: {user_prompt}\nAI:",
            "parameters": {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
            },
            "options": {"wait_for_model": True}
        }
        # payload = {
        #     "inputs": f"{qa_prompt} {user_prompt}",
        #     "parameters": {
        #         "max_new_tokens": request.max_new_tokens,
        #         "temperature": request.temperature,
        #     },
        #     "options": {"wait_for_model": True}
        # }
    else:
        general_prompt_template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.  

Current conversation:
{history}
Human: {input}
AI:"""
        general_prompt = general_prompt_template.format(history="", input=user_prompt)
        payload = {
            "inputs": general_prompt,
            "parameters": {
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
            },
            "options": {"wait_for_model": True}
        }
    headers = {
        "Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"
    }

    # model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    API_URL = f"https://api-inference.huggingface.co/models/{model_id}"

    response = requests.post(API_URL, headers=headers, json=payload)
    print("Hooooooy! response is: ", response)

    # question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

    # rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # store = {}

    # def get_session_history(session_id: str) -> BaseChatMessageHistory:
    #     if session_id not in store:
    #         store[session_id] = ChatMessageHistory()
    #     return store[session_id]

    # conversational_rag_chain = RunnableWithMessageHistory(
    #     rag_chain,
    #     get_session_history,
    #     input_messages_key="input",
    #     history_messages_key="chat_history",
    #     output_messages_key="answer",
    # )
    # response = conversational_rag_chain.invoke(
    #     {"input": question},
    #     config={
    #         "configurable": {"session_id": "abc123"}
    #     },  # constructs a key "abc123" in `store`.
    # )
    # response = response["answer"].split('Assistant: ')[-1].strip()


    try:
        data = response.json()
        if 'error' in data:
            return {"response": f"Error: {data['error']}"}

        generated_text = data[0]["generated_text"]
        if context != None or context != "":
            print("Attention!!!!!!!!!!!! You got here!")
            # answer = generated_text[len(context_docs + user_prompt):].strip()
            ai_response_start = generated_text.find("AI:") + len("AI:")
            answer = generated_text[ai_response_start:].strip()

            # # Extract the last paragraph from the response
            # last_paragraph = generated_text.split('\n')[-1].strip()
            # answer = last_paragraph
        else:
            print("Attention!!!!!!!!!!!! You DID NOT get there!")
            answer = generated_text.strip()
    except Exception as e:
        answer = f"Error processing response: {str(e)}"
        print("Error response content:", response.text)

    return {"response": answer}

# @app.post("/query")
# def query_model(request: QueryRequest):
#     user_prompt = request.prompt
#     print("user_prompt = ", user_prompt)

#     logging.info(f"Received request data: {request}")
#     response_data = {"response": "Hello, World!"}  # Replace with your actual logic
#     logging.info(f"Sending response data: {response_data}")

#     context = None
#     database = None

#     # Check for any files in the directory
#     file_paths = [os.path.join(FILES_DIR, f) for f in os.listdir(FILES_DIR) if f.endswith(".pdf")]
#     if file_paths:
#         print("Found PDF files:", file_paths)
#         docs = rag_util.load_and_split_pdfs(file_paths)
#         DB = rag_util.PGVectorDb(docs=docs, embedding_function=encoder.embedding_function)
#         context = DB.similarity_search(user_prompt, k=request.k)
#         database = DB.get_db()
#     else:
#         print("No PDF files found in directory:", FILES_DIR)



#     answer = model.generate(
#         user_prompt,
#         db=database,
#         context=context,
#         max_new_tokens=request.max_new_tokens,
#         temperature=request.temperature,
#         k=request.k,
#     )

#     return {"response": answer}








@app.post("/uploadfile/")
async def upload_file(uploaded_file: UploadFile = File(...)):
    file_path = os.path.join(FILES_DIR, uploaded_file.filename)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.file.read())
    return {"filename": uploaded_file.filename}

@app.delete("/deletefile/{file_name}")
async def delete_file(file_name: str):
    file_path = os.path.join(FILES_DIR, file_name)
    print(f"Attempting to delete file: {file_path}")  # Log the file path
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"File {file_path} deleted successfully.")  # Log successful deletion
        return JSONResponse(status_code=200, content={"status": "success"})
    else:
        print(f"File {file_path} not found.")  # Log file not found
        raise HTTPException(status_code=404, detail="Not Found")


@app.on_event("shutdown")
async def shutdown_event():
    # Remove all files in the directory on shutdown
    if os.path.exists(FILES_DIR):
        shutil.rmtree(FILES_DIR)
    print("Cleaned up uploaded files")






# @app.post("/uploadfile/")
# async def upload_file(uploaded_file: UploadFile = File(...)):
#     file_path = os.path.join(FILES_DIR, uploaded_file.filename)
#     os.makedirs(FILES_DIR, exist_ok=True)
#     with open(file_path, "wb") as f:
#         f.write(await uploaded_file.read())
#     print("File uploaded successfully:", uploaded_file.filename)

#     return {"filename": uploaded_file.filename}



# @app.post("/query")
# def query_model(request: QueryRequest):
#     user_prompt = request.prompt
#     print("user_prompt = ", user_prompt)

#     logging.info(f"Received request data: {request}")
#     response_data = {"response": "Hello, World!"}  # Replace with your actual logic
#     logging.info(f"Sending response data: {response_data}")

#     context = None
#     database = None

#     # Check for any files in the directory
#     file_paths = [os.path.join(FILES_DIR, f) for f in os.listdir(FILES_DIR) if f.endswith(".pdf")]
#     if file_paths:
#         print("Found PDF files:", file_paths)
#         docs = rag_util.load_and_split_pdfs(file_paths)
#         DB = rag_util.PGVectorDb(docs=docs, embedding_function=encoder.embedding_function)
#         context = DB.similarity_search(user_prompt, k=request.k)
#         database = DB.get_db()
#     else:
#         print("No PDF files found in directory:", FILES_DIR)

#     model_name = "meta-llama/Meta-Llama-3-8B-Instruct"


#     answer = model.generate(
#         user_prompt,
#         db=database,
#         context=context,
#         max_new_tokens=request.max_new_tokens,
#         temperature=request.temperature,
#         k=request.k,
#     )

#     ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
#     # model_name = "huawei-noah/TinyBERT_General_4L_312D"
#     # model = AutoModel.from_pretrained(model_name, token=ACCESS_TOKEN)
#     # tokenizer = AutoTokenizer.from_pretrained(model_name, token=ACCESS_TOKEN)

#     model_name = "gpt2"  # Use a model that supports text generation
#     model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=ACCESS_TOKEN)
#     tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=ACCESS_TOKEN)

#     # model = BertLMHeadModel.from_pretrained(model_name, use_auth_token=ACCESS_TOKEN)
#     # tokenizer = BertTokenizer.from_pretrained(model_name, use_auth_token=ACCESS_TOKEN)
#     # # Tokenize input text
#     # input_ids = tokenizer.encode(user_prompt, return_tensors='pt')
#     inputs = tokenizer.encode(user_prompt, return_tensors='pt')
#     print("Shape of input_ids tensor:", inputs.shape)
#     print(" input_ids tensor is :", inputs["input_ids"])
#     # # Generate text
#     # outputs = model.generate(input_ids=input_ids, max_length=50)
#     outputs = model.generate(input_ids=inputs["input_ids"], max_length=50)

#     # inputs = tokenizer(user_prompt, return_tensors="pt")

#     # # Model apply
#     # outputs = model(**inputs)
#     answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

#     # answer = "This is the funky answer!"
#     return {"response": answer}

