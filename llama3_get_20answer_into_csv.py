###########################################################################################
## Pre-requirements
# 1. Make sure all libraries are installed
# 2. Make sure the file 'GCME HR MANUAL.pdf' is in the same dir as the python file
# 3. Make sure to create a '.env' file that contains the postgres sql password as 'postgres_pass = 'Your_Password'
# 4. The postgress link is follow, revise if you need: 
#    user = 'postgres',password = postgres_pass, host = 'localhost:5432', database = 'vector_db'
#    CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}/{database}" (in line:201)
# 3. Change the model name accordingly (in line:98)
# 4. Change the parameters for pipeline accordingly (in line:236)
# 5. The result will create a csv file in folder 'GCME_HR_MANUAL_chat_data' under the same dir as the python file, 
#    change the folder name as you wish (in line:229)

from langchain.vectorstores.pgvector import PGVector
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings import SentenceTransformerEmbeddings
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain import HuggingFacePipeline
from langchain.chains.retrieval_qa.base import RetrievalQA
import time
import pandas as pd
from dotenv import load_dotenv
import os
import re
from langchain_core.documents.base import Document
from langchain_core.prompts import PromptTemplate
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm

load_dotenv()
postgres_pass = os.getenv('postgres_pass')

def get_texts(pdf_file, chunk_size=1024,chunk_overlap=64):
    # Load the PDF file from current working directory
    loader = PyPDFLoader(pdf_file)

    # Split the PDF into Pages
    pages = loader.load_and_split()
    def normalize_spaces(doc):
        # Replace multiple occurrences of '\xa0' or spaces with a single space
        normalized_page_content = re.sub(r'\s+', ' ', doc.page_content.replace('‐‐', ' ').replace('--', ' '))
        return Document(metadata=doc.metadata,page_content=normalized_page_content)
    
    # Define chunk size, overlap and separators
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=['\n\n', '\n', '(?=>\. )', '\xa0', ' ', '']
    )

    # Split the pages into texts as defined above
    normalized_pages = [normalize_spaces(page) for page in pages]
    texts = text_splitter.split_documents(normalized_pages)
    return texts, chunk_size, chunk_overlap

def get_db(texts,model_name,user,password,host,database):
    # Load embeddings from HuggingFace
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2") # Rank 83 0.41 gb memory usage
    # embeddings = HuggingFaceEmbeddings(model_name="mixedbread-ai/mxbai-embed-large-v1") # Rank 11 1.25 gb memory usage
    # embeddings = HuggingFaceEmbeddings(model_name="WhereIsAI/UAE-Large-V1") # Rank 12 1.25 gb memory usage
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    CONNECTION_STRING = f"postgresql+psycopg2://{user}:{password}@{host}/{database}"
    # "postgresql+psycopg2://user_name:pass_word@host_name:port/database_name"
    COLLECTION_NAME = 'pdf_collection_vectors'
    db = PGVector.from_documents(embedding=embeddings, 
                                 documents=texts, 
                                 collection_name = COLLECTION_NAME,
                                 connection_string=CONNECTION_STRING,
                                 use_jsonb=True,
                                 pre_delete_collection=True,
                                )
    return db, model_name

def get_model():
#     #Set model
#     model = "tiiuae/falcon-7b-instruct"
#     # If GPU available use it
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     # Load tokenizer
#     tokenizer = AutoTokenizer.from_pretrained(model)
#     # Load model
#     model = AutoModelForCausalLM.from_pretrained(
#         model,
# #         trust_remote_code=True,
#     #     load_in_8bit=True,
#         quantization_config=BitsAndBytesConfig(load_in_8bit=True),
#         device_map='auto'
#     )
#     # Set to eval mode
#     model.eval()
    
    
    #Set model
    model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    # If GPU available use it
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
    #         trust_remote_code=True,
    #     load_in_8bit=True,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map='auto'
    )
    # Set to eval mode
    model.eval()
    return model, tokenizer, device

# tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")
# device = "cuda" if torch.cuda.is_available() else "cpu"

def get_qa(db,tokenizer,device,k=3,search_type="similarity",max_length=1200,):
    # Create a pipline
#     generate_text = pipeline(task="text-generation", model=model, tokenizer=tokenizer, 
#                              trust_remote_code=True, max_new_tokens=100, 
#                              repetition_penalty=1.1, model_kwargs={"device_map": "auto", 
#                               "max_length": max_length, "temperature": 0.01, "torch_dtype":torch.bfloat16, 'device': device}
#     )

    generate_text = pipeline(
        "text-generation",
        model=model,
        model_kwargs={"torch_dtype": torch.bfloat16,"max_length": 1200, "temperature": 0.01, },
        device_map='auto',
        tokenizer=tokenizer,
        max_new_tokens=500,

    )
    
    # LangChain HuggingFacePipeline set to our transformer pipeline
    llm = HuggingFacePipeline(pipeline=generate_text)
    
    prompt_template = """You are an intelligent chatbot.
Provide a detailed response to the question based only on the provided context.
Refrain from inventing any information not explicitly stated in the context.
If uncertain, simply state that you do not know instead of speculating.
If you do not find the answer in the context, please state 'I can not find the answer from the uploaded file.'.

{context}

Question: {question}
Answer:"""
    prompt = PromptTemplate(template=prompt_template, input_variables=["context","question"])
    
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", 
                                 retriever=db.as_retriever(search_kwargs={"k": k,"search_type": search_type}), 
                                     # search_type: "similarity" or "mmr"
                                 chain_type_kwargs={"prompt": prompt}, # This is how to add the prompt in the RetrievalQA
                                 return_source_documents=True,
                                 verbose=False,
    )
    return qa, search_type, k

def get_answer(qa,query):
    start = time.time()
    result = qa({"query": query})
    answer = result["result"].split('Answer: ')[-1].strip()
#     print(f'Answer: {answer}')
    ref_pages = []
    for source in result['source_documents']:
        ref_pages.append(source.metadata['page'])
#     print(f'Reference pages: {ref_pages}')
    k = len(ref_pages)
    end = time.time()
    duration = end-start
#     print(f'Took {duration} secs')
    return answer, k, ref_pages, duration
    
question_list = ['What is the procedure for requesting time off?',
'How should employees report workplace injuries or accidents?',
'What is the policy regarding internet and email usage during work hours?',
'Can employees bring personal devices (laptops, tablets, etc.) to work?',
'What is the policy on dress code and grooming standards?',
'Are there any guidelines for social media usage related to the company?',
'How should employees handle confidential information and client data?',
'What is the procedure for requesting reimbursement for work-related expenses?',
'Are there any guidelines for working remotely or telecommuting?',
"What is the company's policy on attendance and punctuality?",
'How should employees address conflicts of interest?',
'Is there a policy regarding the use of company-owned vehicles?',
'What is the procedure for requesting parental leave or other types of leave?',
'Are there any guidelines for workplace conduct and professionalism?',
'What is the policy regarding smoking and tobacco use on company premises?',
'How should employees handle conflicts or disputes with colleagues or supervisors?',
'Are there any guidelines for accepting gifts or gratuities from clients or vendors?',
'What is the policy regarding alcohol consumption at company events or functions?',
'How should employees handle requests for references or recommendations?',
'What is the procedure for reporting harassment or discrimination in the workplace?'
]

model, tokenizer, device = get_model()

def save_data(tokenizer=tokenizer,device=device,chunk_size=1024,chunk_overlap=64,model_name="sentence-transformers/all-mpnet-base-v2",k=3,search_type="similarity"):    
    start_count = time.time()

    texts, chunk_size, chunk_overlap = get_texts("GCME HR MANUAL.pdf", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    db, model_name = get_db(texts,model_name,'postgres',postgres_pass,'localhost:5432','vector_db')
    qa, search_type, k = get_qa(db,tokenizer=tokenizer,device=device,k=k,search_type=search_type,max_length=1200)
    chat_data_list = ['query',
                      'answer',
                      'search_type',
                      'chunk_size',
                      'chunk_overlap',
                      'embedding_model',
                      'k',
                      'ref_pages',
                      'time_consumed'
                     ]

    chat_data_dict = {field: [] for field in chat_data_list}
    for query in tqdm(question_list, desc="Processing queries"):
        answer, k, ref_pages, duration = get_answer(qa,query)
        chat_data_dict['query'].append(query)
        chat_data_dict['answer'].append(answer)
        chat_data_dict['search_type'].append(search_type)
        chat_data_dict['chunk_size'].append(chunk_size)
        chat_data_dict['chunk_overlap'].append(chunk_overlap)
        chat_data_dict['embedding_model'].append(model_name)
        chat_data_dict['k'].append(k)
        chat_data_dict['ref_pages'].append(ref_pages)
        chat_data_dict['time_consumed'].append(duration)
        
    df = pd.DataFrame(chat_data_dict)
    model_name_new = model_name.replace('/','--')
    output_dir = 'GCME_HR_MANUAL_chat_data'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f'{output_dir}/{model_name_new}_{chunk_size}_{search_type}_{k}_GCMEHRMANUAL.csv',index=False)
    print(f'Totally took {time.time()-start_count} secs')
    return df
    
if __name__ == "__main__":
    df = save_data(chunk_size=1024,chunk_overlap=64,model_name="sentence-transformers/all-mpnet-base-v2",k=3,search_type='similarity')
