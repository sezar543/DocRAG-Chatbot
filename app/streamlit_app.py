import os
import streamlit as st
import requests
from datetime import datetime
import json
# from shared_memory import get_memory
import uuid
import atexit
from dotenv import load_dotenv
import shutil

# Load environment variables from .env file
load_dotenv()

os.environ["FASTAPI_HOST"] = "localhost"

st.title("LLM Chatbot RAG Assistant")
print("The print statement below st.title. The code is read from top!")

# API endpoints
UPLOAD_URL = os.getenv("UPLOAD_URL", "http://localhost:8000/uploadfile/")
DELETE_URL = os.getenv("DELETE_URL", "http://localhost:8000/deletefile/")
CHAT_URL = os.getenv("CHAT_URL", "http://localhost:8000/query")
backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")

# Ensure the variables are loaded
if not all([UPLOAD_URL, DELETE_URL, CHAT_URL]):
    raise ValueError("One or more environment variables are not set properly.")

print("UPLOAD_URL is")
print(UPLOAD_URL)

# # Directory to store uploaded files
FILES_DIR = "./uploaded_files"
# FILES_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files"))
os.makedirs(FILES_DIR, exist_ok=True)

# Function to get or generate session ID
def get_or_generate_session_id():
    # Check if session ID is already stored in session state or query params
    if 'session_id' not in st.session_state:
        # Check if session ID is stored in query params
        stored_session_id = st.query_params.get("session_id")

        if stored_session_id:
            st.session_state.session_id = stored_session_id
        else:
            # Generate a new session ID if not found
            st.session_state.session_id = str(uuid.uuid4())
            # Store the generated session ID in query params
            st.query_params.session_id = st.session_state.session_id

    return st.session_state.session_id

session_id = get_or_generate_session_id()

if 'session_id' not in st.session_state:
    st.session_state['session_id'] = session_id

# Initialize the previous uploaded file state
if 'previous_uploaded_file' not in st.session_state:
    st.session_state.previous_uploaded_file = None

session_folder = f"./uploaded_files/{st.session_state.session_id}"
TRACKING_FILE = f"./uploaded_files/{st.session_state.session_id}/tracking.json"


# session_id = st.session_state['session_id']
# print("session_id = ", session_id)



# File to store session state data
STATE_FILE = "session_state.json"
# TRACKING_FILE = './uploaded_files/tracking.json'

# Get the model names from environment variables
llm_model_id = os.getenv("LLM_model_id")
model_encoder = os.getenv("model_encoder")

# Display the model names in the sidebar
st.sidebar.title("Model Information")
st.sidebar.write(f"LLM Model: {llm_model_id}")
st.sidebar.write(f"Encoder Model: {model_encoder}")

def reset_conversation():
    st.session_state.messages = []
    st.session_state.responses = []
    st.session_state.uploaded_files = []
    st.session_state.initialized = False
  
# Function to save tracking.json
def save_tracking_json():
    tracking_path = os.path.join(session_folder, 'tracking.json')
    with open(tracking_path, 'w') as f:
        json.dump({'uploaded_files': st.session_state.uploaded_files}, f)

# Function to read tracking.json
def read_tracking_json():
    tracking_path = os.path.join(session_folder, 'tracking.json')
    if os.path.exists(tracking_path):
        with open(tracking_path, 'r') as f:
            return json.load(f)
    return {}

def print_tracking_json():
    print("Content of tracking.json:")
    tracking_path = os.path.join(session_folder, 'tracking.json')
    try:
        if os.path.exists(tracking_path):
            with open(tracking_path, 'r') as f:
                data = json.load(f)
                print(json.dumps(data, indent=4))
        else:
            print("tracking.json is empty ")
    except FileNotFoundError:
        print("tracking.json not found.")

def read_state():
    if os.path.exists(TRACKING_FILE):
        with open(TRACKING_FILE, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

# Load state from tracking.json if available
def load_state():
    state = read_tracking_json()
    if isinstance(state, dict):
        st.session_state.uploaded_files = state.get("uploaded_files", [])
    else:
        st.session_state.uploaded_files = state

# Initialize session state
if 'initialized' not in st.session_state:
    print("Did it initialize?")
    load_state()
    st.session_state.initialized = True

# State to track uploaded files
# if 'uploaded_files' not in st.session_state:
#     st.session_state.uploaded_files = []

if 'file_uploader_key' not in st.session_state:
    st.session_state.file_uploader_key = str(uuid.uuid4())

if 'responses' not in st.session_state:
    print("responses not in st.session_state")
    st.session_state.responses = []

# Initialize chat history
if "messages" not in st.session_state:
    print("messages not in st.session_state ", st.session_state)
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    print("Display chat messages from history on app rerun")
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to load the list of uploaded files from JSON file
def read_uploaded_files():
    state = read_state()
    print("inside read_uploaded_files, state is ", state)
    return state.get("uploaded_files", [])

# Function to save the list of uploaded files to JSON file
def save_uploaded_files(uploaded_files):
    # with open(TRACKING_FILE, 'w') as f:
    #     json.dump(file_list, f)
    # with open("session_state.json", "w") as f:
    #     json.dump({"uploaded_files": uploaded_files}, f)
    state = read_state()
    state["uploaded_files"] = uploaded_files
    print("Inside the method save_uploaded_files: variable state is ", state)
    save_state(state)

def save_file(uploaded_file):
    print("Did we get into save_file ?!")
    for _ in range(3):  # Retry mechanism
        # """Helper function to save documents to disk via FastAPI."""
        file = {"uploaded_file": uploaded_file}
        headers = {"X-Session-ID": session_id}
        response = requests.post(UPLOAD_URL, files=file, headers=headers)
        if response.status_code == 200:
            file_info = response.json()
            st.session_state.uploaded_files.append(file_info)
            save_tracking_json()
            print("After save_file, st.session_state.uploaded_files is ", st.session_state.uploaded_files)
            save_uploaded_files(st.session_state.uploaded_files)

            return
        else:
            st.error("File upload failed. Retrying...")
    st.error("File upload failed after multiple attempts.")
    st.error(f"Error: {response.status_code}, {response.text}")

    print("After save_file, st.session_state.uploaded_files is ", st.session_state.uploaded_files)

def delete_file(file_info):
    print("Inside delete_file: file_info is ", file_info)
    try:
        # Send delete request to FastAPI
        # headers = {'x-session-id': session_id}
        headers = {'x-session-id': st.session_state.session_id}
        print("Inside delete_file: st.session_state", st.session_state)
        print("Inside delete_file: session_id ", session_id)
        url = f"{DELETE_URL}{file_info['name']}"
        response = requests.delete(url, headers=headers)
        print("Delete_file: url = ", url)

        if response.status_code == 200:
            print("response.status_code == 200!!!!")
            print("st.session_state.uploaded_files = ", st.session_state.uploaded_files)
            st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f["name"] != file_info["name"]]            
            print("st.session_state.uploaded_files is ", st.session_state.uploaded_files)
            uploaded_files = read_uploaded_files()
            uploaded_files = [f for f in uploaded_files if f["name"] != file_info["name"]]
            save_uploaded_files(uploaded_files)
            return True
        else:
            print(f"Error deleting file {file_info['path']}: {response.text}")
            return False
    except Exception as e:
        print(f"Error deleting file {file_info['path']}: {e}")
        return False

def cleanup_files():
    print("Inside cleanup_files")
    uploaded_files = read_uploaded_files()  # Read uploaded files state
    new_uploaded_files = []
    
    for file_info in uploaded_files:
        if 'session_id' in st.session_state and st.session_state.session_id in file_info['path']:
            folder_to_delete = os.path.dirname(file_info['path'])
            try:
                shutil.rmtree(folder_to_delete)
                print(f"Deleted session folder: {folder_to_delete}")
            except Exception as e:
                print(f"Error deleting session folder {folder_to_delete}: {str(e)}")
        else:
            # Keep non-matching uploaded files in the state
            new_uploaded_files.append(file_info)
    # Save the updated uploaded files state
    save_uploaded_files(new_uploaded_files)
    
    if 'uploaded_files' in st.session_state:
        st.session_state.uploaded_files = []
        print("Method cleanup_files: st.session_state.uploaded_files is ", st.session_state.uploaded_files)


def cleanup():
    print("Inside cleanup")
    if os.path.exists(session_folder):
        shutil.rmtree(session_folder)

##########
def save_state(state=None):
    print("Iside save_state: state is", state)
    if state is None:
        state = {
            "uploaded_files": st.session_state.uploaded_files,
            "responses": st.session_state.responses
        }

    try:
        if os.path.exists(TRACKING_FILE):
            print("Debugging for Saving state: ", state)  # Debugging print
            with open(TRACKING_FILE, "w") as f:
                json.dump(state, f)
            with open(TRACKING_FILE, "r") as f:
                tracking_data = json.load(f)
            uploaded_files = tracking_data.get("uploaded_files", [])
            print("Iside save_state: State saved:", st.session_state.uploaded_files, st.session_state.responses)  # Debugging print
            print("TRACKING_FILE:", uploaded_files)
        else:
            print(f"Tracking file not found. Not creating new tracking file.")
    except Exception as e:
        print(f"Error saving state: {e}")



def clear_uploaded_files():
    """Helper function to delete all uploaded files on disk."""
    if os.path.exists(FILES_DIR):
        for file_name in os.listdir(FILES_DIR):
            file_path = os.path.join(FILES_DIR, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Error deleting file {file_path}: {e}")

# Function to append new messages
def append_message(message):
    st.session_state.messages.append(message)

# Register the cleanup function to be called on exit
# atexit.register(cleanup_files)
atexit.register(cleanup)


# Load previously uploaded files on app start
uploaded_files = read_uploaded_files()
print("uploaded_files is ", uploaded_files)
# for file_info in uploaded_files:
#     if session_id in file_info['path']:
#         st.session_state.uploaded_files.append(file_info)
#         st.sidebar.write(file_info['name'])


with st.sidebar:
    st.write(f"Session ID: {session_id}")
    max_new_tokens = st.number_input("max_new_tokens", 128, 4096, 512)
    k = st.number_input("k", 1, 10, 3)
    temperature = st.number_input("temperature", 0.01, 0.99, 0.01)

    # File uploader
    print("Before uploaded_file: Tracking.josn is:")
    print_tracking_json()
    print("Before uploaded_file: st.session_state.uploaded_files is", st.session_state.uploaded_files)
    uploaded_file = st.file_uploader("Upload PDFs for context", type=["PDF", "pdf"], accept_multiple_files=False, key=st.session_state.file_uploader_key)
    # uploaded_file = st.file_uploader("Upload PDFs for context", type=["PDF", "pdf"], accept_multiple_files=False, key=st.session_state.file_uploader_key, label_visibility="hidden") 
    print("Inside st.side.bar: ", uploaded_file)
    
    # Detect file removal
    if uploaded_file is None and st.session_state.previous_uploaded_file:
        # A file was removed, delete the file from the system
        to_delete = None
        for file_info in st.session_state.uploaded_files:
            if file_info['name'] == st.session_state.previous_uploaded_file.name:
                delete_file(file_info)
                to_delete = file_info
                break
        print("After delete_file: st.session_state.uploaded_files: ", st.session_state.uploaded_files)
        if to_delete and to_delete in st.session_state.uploaded_files:
            st.session_state.uploaded_files.remove(to_delete)
            save_uploaded_files(st.session_state.uploaded_files)
        st.session_state.previous_uploaded_file = None

    
     # Handle file uploads
    # if uploaded_file:
    if uploaded_file:
        print("Inside: if uploaded_file... ")
        # Clear previous files from the sidebar and delete them via FastAPI
        if st.session_state.uploaded_files:
            for file_info in st.session_state.uploaded_files:
                print("This file is already uploaded: ", file_info)
                delete_file(file_info)
            st.session_state.uploaded_files = []

        # file_path = f"./uploaded_files/{st.session_state.session_id}/{uploaded_file.name}"
        file_path = os.path.join(session_folder, uploaded_file.name)
        file_info = {'name': uploaded_file.name, 'path': file_path}
        print("Inside with st.sidebar: file_info is", file_info)
        print("Inside with st.sidebar: st.session_state.uploaded_files is", st.session_state.uploaded_files)

        # Check if the file already exists in session state
        if file_info not in st.session_state.uploaded_files:
            os.makedirs(session_folder, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(uploaded_file.getbuffer())
             
            save_file(uploaded_file)


        st.session_state.previous_uploaded_file = uploaded_file


    print("uploaded_files in st.session_state.uploaded_files are ", st.session_state.uploaded_files)
    # Display and handle file removals
    if st.session_state.uploaded_files:
        file_info = st.session_state.uploaded_files[-1]
        print("side_bar: file_info = ", file_info)
        print("st.session_state.uploaded_files: ", st.session_state.uploaded_files)
        file_name = file_info['name']

        file_path = file_info['path']
        st.write(f"Uploaded file: {file_name}")
        st.write(f"File path: {file_path}")
    else:
        st.write("No files uploaded yet.")



    for idx, file_info in enumerate(st.session_state.uploaded_files):
        # st.write(f"Uploaded file: {file_info['name']}")
        if st.button(f"Remove {file_info['name']}", key=f"remove_{idx}"):
            if delete_file(file_info):
                st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f["session_id"] != file_info["session_id"] and f["name"] != file_info["name"]]
                print("st.session_state.file_uploader_key is ", st.session_state.file_uploader_key)
                st.session_state.file_uploader_key = str(uuid.uuid4())  # Change the key to reset the uploader
                print("st.session_state.file_uploader_key is ", st.session_state.file_uploader_key)
                save_tracking_json()
                print("After delete_file: st.session_state.uploaded_files = ", st.session_state.uploaded_files)
                print("After delete file: Tracking.josn is:")
                print_tracking_json()
                st.rerun()
            else:
                st.error(f"Failed to delete file {file_info['name']}")


memory_reset_trigger = False
# Accept user input
if prompt := st.chat_input("Ask me anything!"):    
    st.session_state.messages.append({"role": "user", "content": prompt})
    # append_message({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    
    payload = {"prompt": prompt, "max_new_tokens": max_new_tokens, "k": k, "temperature": temperature}
    headers = {"X-Session-ID": session_id}
    with st.chat_message("assistant"):
        with st.spinner('Generating answer...'):
            response = requests.post(
                CHAT_URL,
                headers=headers,
                json=payload
            )

            if response.status_code == 200:
                response_data = response.json()
                answer = response_data["response"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                # append_message({"role": "assistant", "content": answer})
                save_state()  # Save state after modifying responses

            else:
                st.write(f"Error: {response.status_code}")
            
            print("Response content:", response.content)

    save_state()



if st.button("Reset Conversation", on_click=reset_conversation):
    headers = {"X-Session-ID": session_id}
    payload = {"session_id": session_id, "k": k}
    print("url for reset conversation is ", f"{backend_url}/reset_conversation")
    response = requests.post(f"{backend_url}/reset_conversation", json=payload, headers=headers)

    if response.status_code == 200:
        st.success("Conversation reset successfully!")
    else:
        st.error("Failed to reset the conversation.")

# Ensure proper cleanup on exit
@st.cache_resource()
def get_cleanup_tracker():
    return []

cleanup_tracker = get_cleanup_tracker()

if st.session_state.session_id not in cleanup_tracker:
    cleanup_tracker.append(st.session_state.session_id)
    atexit.register(cleanup_files)

