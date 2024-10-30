import os
import glob
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
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores.pgvector import PGVector
from transformers import AutoTokenizer
from shared_memory import get_memory, reset_memory
import logging


# import logging
# logging.basicConfig(level=logging.INFO)

# from transformers import logging as transformers_logging
# transformers_logging.set_verbosity_info()

load_dotenv()

ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN") # reads .env file with HUGGINGFACEHUB_API_TOKEN=<your hugging face access token>

# model_id = "mistralai/Mistral-7B-Instruct-v0.2"
LLM_model_id = os.getenv("LLM_model_id")
if not LLM_model_id:
    raise ValueError("The LLM_MODEL_ID environment variable is not set.")
print(f"model.py Loading model with ID: {LLM_model_id}")

# model_encoder = os.getenv("model_encoder")

# memory = ConversationBufferWindowMemory(k=3) # memory_key="chat_history",input_key="human_input", human_prefix="Human",ai_prefix="AI Assistant",

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

class ChatModel:
    def __init__(self, LLM_model_id: str = LLM_model_id, memory = [], device="cuda"):
        logging.debug("Loading model with ID: %s", LLM_model_id)

        self.LLM_model_id = LLM_model_id
        # self.memory = memory
        self.device = device
        self.memory_store = {}

        # Download model to cache if not already cached
        # self.ensure_model_in_cache()

        self.tokenizer = AutoTokenizer.from_pretrained(
            LLM_model_id, 
            cache_dir=CACHE_DIR, 
            token=ACCESS_TOKEN
        )

        # self.model.generation_config.pad_token_id = tokenizer.pad_token_id
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # If GPU, use following: 
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

        self.model = AutoModelForCausalLM.from_pretrained(
            LLM_model_id,
            device_map="auto",
            # load_in_8bit_fp32_cpu_offload=True,  #added            
            quantization_config=quantization_config,
            cache_dir=CACHE_DIR,
            token=ACCESS_TOKEN,
            pad_token_id=self.tokenizer.eos_token_id  # Explicitly set pad_token_id
        )

    def to_dict(self):
        return {
            "LLM_model_id": self.LLM_model_id,
            "device": self.device,
            # Add other necessary attributes here
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            LLM_model_id=data["LLM_model_id"],
            device=data["device"],
            # Initialize other necessary attributes here
            )

        logging.debug("Model and tokenizer loaded successfully")



    def get_memory(self, user_id, k):
        print("k inside get_memory method = ", k)
        if user_id not in self.memory_store or self.memory_store[user_id].k != k:
            print("user_id found in self.memory_store!!")
            self.memory_store[user_id] = ConversationBufferWindowMemory(k=k)
        print("self.memory_store[user_id] = ", self.memory_store[user_id])
        return self.memory_store[user_id]

    def reset_memory(self, user_id, k):
        print("k inside reset_memory method = ", k)
        self.memory_store[user_id] = ConversationBufferWindowMemory(k=k)

    # def check_memory(self, memory_reset_trigger=False):
    #     if memory_reset_trigger:
    #         self.memory = []  # Reset memory

    def check_memory(self, user_id: str, memory_reset_trigger: bool = False, k=3):
        print("user_id inside reset_conversation of fastapi=", user_id)
        print("k inside check_memory method = ", k)
        if memory_reset_trigger:
            if user_id in self.memory_store:
                print("Memory is being reset!")
                self.reset_memory(user_id, k)
        return {"status": "memory_reset" if memory_reset_trigger else "memory_checked"}


    # def ensure_model_in_cache(self):
    #     # Check if model files are already cached
    #     cache_model_dir = os.path.join(CACHE_DIR, f"models--{self.LLM_model_id.replace('/', '--')}")
    #     cached_files = glob.glob(os.path.join(cache_model_dir, "*"))
    #     print(f"Cache directory: {cache_model_dir}")
    #     print(f"Cached files: {cached_files}")

    #     if not cached_files:
    #         print("Model files not found in cache. Downloading model.")
    #         hf_hub_download(
    #             repo_id=self.LLM_model_id,
    #             filename="pytorch_model.bin",
    #             cache_dir=CACHE_DIR,
    #             token=self.access_token
    #         )
    #         hf_hub_download(
    #             repo_id=self.LLM_model_id,
    #             filename="config.json",
    #             cache_dir=CACHE_DIR,
    #             token=self.access_token
    #         )
    #         hf_hub_download(
    #             repo_id=self.LLM_model_id,
    #             filename="tokenizer_config.json",
    #             cache_dir=CACHE_DIR,
    #             token=self.access_token
    #         )
    #         hf_hub_download(
    #             repo_id=self.LLM_model_id,
    #             filename="vocab.txt",
    #             cache_dir=CACHE_DIR,
    #             token=self.access_token
    #         )
    #         # Add any additional files needed for your model
    #     else:
    #         print(f"Model files found in cache: {cached_files}")


    # def check_cache(self):
    #     # Path to cached model files
    #     cache_model_dir = os.path.join(CACHE_DIR, self.LLM_model_id.replace('/', '_'))
    #     cached_files = glob.glob(os.path.join(cache_model_dir, "*"))

    #     print(f"Cache directory: {cache_model_dir}")
    #     print(f"Cached files: {cached_files}")

    #     if cached_files:
    #         print(f"Model files found in cache: {cached_files}")
    #     else:
    #         print("Model files not found in cache. Model will be downloaded.")

    async def generate(self, user_id: str, question: str, db=None, context: str = None, max_new_tokens: int = 250, temperature: float = 0.01, max_length: int = 1200, k: int = 3, ):
        self.generate_text = pipeline(
            "text-generation",
            model=self.model,
            model_kwargs={"torch_dtype": torch.bfloat16,"max_length": max_length, "temperature": temperature, },
            device_map="auto",
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
        )

        # LangChain HuggingFacePipeline set to our transformer pipeline
        self.llm = HuggingFacePipeline(pipeline=self.generate_text)

        memory = self.get_memory(user_id, k)
        print("memory is ", memory)
        
        if context == None or context == "":
            prompt_template = """The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.  

Current conversation:
{history}
Human: {input}
AI:"""

            self.prompt = PromptTemplate(template=prompt_template, input_variables=["history","input"])

            llm_chain = ConversationChain(prompt=self.prompt,llm=self.llm,memory=memory,) # 
            response = llm_chain.predict(input=question) # return_only_outputs=True
            response = response.split(question)[1].strip().split('Human:')[0].strip()[4:]
            return response
                    
        else:
            prompt_template = """You are an intelligent chatbot. 
Provide a detailed response to the following question based only on the provided context. 
Refrain from inventing any information not explicitly stated in the context. 
If uncertain, simply state that you do not know instead of speculating. 
If you do not find the answer in the context, say 'I can not find the answer from the uploaded file.'.
Context: {context}.
Question: {question}
Answer:"""
            self.prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
            
            qa = RetrievalQA.from_chain_type(llm=self.llm, chain_type="stuff", 
                        retriever=db.as_retriever(search_kwargs={"k": k,"search_type": "similarity"}), 
                            # search_type: "similarity" or "mmr"
                        chain_type_kwargs={"prompt": self.prompt}, # This is how to add the prompt in the RetrievalQA
                        return_source_documents=True,
                        verbose=False,
            )
            response = qa.invoke({"query":question})
            response = response["result"].split('Answer: ')[-1].strip()

#             contextualize_q_system_prompt = """Given a chat history and the latest user question \
# which might reference context in the chat history, formulate a standalone question \
# which can be understood without the chat history. Do NOT answer the question, \
# just reformulate it if needed and otherwise return it as is."""
#             contextualize_q_prompt = ChatPromptTemplate.from_messages(
#                 [
#                     ("system", contextualize_q_system_prompt),
#                     MessagesPlaceholder("chat_history"),
#                     ("human", "{input}"),
#                 ]
#             )
#             history_aware_retriever = create_history_aware_retriever(
#                 self.llm, db.as_retriever(search_kwargs={"k": k,"search_type": "similarity"}), contextualize_q_prompt
#             )

#             qa_system_prompt = """You are an assistant for question-answering tasks. \
# Use the following pieces of retrieved context to answer the question. \
# If you don't know the answer, just say that you don't know. \
# Use three sentences maximum and keep the answer concise.\

# {context}"""
#             qa_prompt = ChatPromptTemplate.from_messages(
#                 [
#                     ("system", qa_system_prompt),
#                     MessagesPlaceholder("chat_history"),
#                     ("human", "{input}"),
#                 ]
#             )


#             question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)

#             rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

#             store = {}


#             def get_session_history(session_id: str) -> BaseChatMessageHistory:
#                 if session_id not in store:
#                     store[session_id] = ChatMessageHistory()
#                 return store[session_id]


#             conversational_rag_chain = RunnableWithMessageHistory(
#                 rag_chain,
#                 get_session_history,
#                 input_messages_key="input",
#                 history_messages_key="chat_history",
#                 output_messages_key="answer",
#             )
#             response = conversational_rag_chain.invoke(
#                 {"input": question},
#                 config={
#                     "configurable": {"session_id": "abc123"}
#                 },  # constructs a key "abc123" in `store`.
#             )
            # response = response["answer"].split('Assistant: ')[-1].strip()
        return response