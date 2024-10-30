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

# def get_session_id():
#     session_state = st.session_state
#     if 'session_id' not in session_state:
#         session_state.session_id = str(uuid.uuid4())
#     return session_state.session_id

# session_id = get_session_id()

# Function to get or generate session ID
def get_or_generate_session_id():
    # Check if session ID is already stored in session state or query params
    if 'session_id' not in st.session_state:
        # Check if session ID is stored in query params
        
        # stored_session_id = st.query_params["session_id"]
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
print("session_id = ", session_id)



# File to store session state data
STATE_FILE = "session_state.json"
# TRACKING_FILE = './uploaded_files/tracking.json'

# # Main section to handle chat and responses
# st.header("Chat with the LLM")

# Get the model names from environment variables
llm_model_id = os.getenv("LLM_model_id")
model_encoder = os.getenv("model_encoder")

# Display the model names in the sidebar
st.sidebar.title("Model Information")
st.sidebar.write(f"LLM Model: {llm_model_id}")
st.sidebar.write(f"Encoder Model: {model_encoder}")

# # Display the list of uploaded files
# st.sidebar.write("Uploaded files:")
# for file_info in st.session_state.get("uploaded_files", []):
#     st.sidebar.write(file_info['name'])

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

# # Function to read the tracking file
# def read_uploaded_files():
#     print("Inside the method read_uploaded_files")
#     # if os.path.exists(TRACKING_FILE):
#     #     with open(TRACKING_FILE, 'r') as f:
#     #         return json.load(f)
#     # # print("TRACKING_FILE is ", TRACKING_FILE)
#     # return []
#     try:
#         with open("tracking.json", "r") as f:
#             return json.load(f)
#     except FileNotFoundError:
#         return []

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
    # response = requests.post(UPLOAD_URL, files={"uploaded_file": uploaded_file})
    # if response.status_code == 200:
    #     file_info = response.json()
    #     st.session_state.uploaded_files.append(file_info)
    #     save_state()
    # else:
    #     st.error("File upload failed.")
    #     st.error(f"Error: {response.status_code}, {response.text}")

# def delete_file(file_name):
#     """Helper function to delete documents from disk via FastAPI."""
#     print("delete_url + file_name = ", f"{DELETE_URL}{file_name}")
#     for _ in range(3):  # Retry mechanism
#         headers = {"X-Session-ID": session_id}
#         response = requests.delete(f"{DELETE_URL}{file_name}", headers=headers)
#         if response.status_code == 200:
#             print(f"File {file_name} deleted successfully.")
#             st.success(f"File {file_name} deleted successfully.")
#             return True
#         else:
#             st.error("Failed to delete file. Retrying...")
#     st.error(f"Failed to delete file {file_name} after multiple attempts.")
#     st.error(f"Error: {response.status_code}, {response.text}")
#     return False

# def delete_file(file_info):
#     print("Inside delete_file: file_info is ", file_info)
#     try:
#         os.remove(file_info["path"])
#         st.session_state.uploaded_files = [f for f in st.session_state.uploaded_files if f["name"] != file_info["name"]]
#         print("st.session_state.uploaded_files is ", st.session_state.uploaded_files)
#         uploaded_files = read_uploaded_files()
#         uploaded_files = [f for f in uploaded_files if f["name"] != file_info["name"]]
#         save_uploaded_files(uploaded_files)
#         return True
#     except Exception as e:
#         print(f"Error deleting file {file_info['path']}: {e}")
#         return False

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

# # Define cleanup function to delete uploaded files
# def cleanup_files():
#     for file_info in uploaded_files:
#         try:
#             os.remove(file_info['path'])
#             print(f"File {file_info['path']} deleted.")
#         except Exception as e:
#             print(f"Error deleting file {file_info['path']}: {e}")

# Define cleanup function to delete uploaded files

# def cleanup_files():
#     print("cleanuo_files: st.session_state is ", st.session_state)
#     if 'uploaded_files' in st.session_state:
#         for file_info in st.session_state.uploaded_files:
#             try:
#                 os.remove(file_info['path'])
#                 print(f"File {file_info['path']} deleted.")
#             except Exception as e:
#                 print(f"Error deleting file {file_info['path']}: {e}")
#         st.session_state.uploaded_files = []

# def cleanup_files():
#     uploaded_files = read_uploaded_files()
#     for file_info in uploaded_files:
#         if file_info.get("session_id") == session_id:
#             delete_file(file_info['name'])
#             uploaded_files.remove(file_info)
#             save_uploaded_files(uploaded_files)
#             print("Method cleanup_fileL st.session_state.uploaded_files is ", st.session_state.uploaded_files)
#             st.session_state.uploaded_files = []
#             print("Method cleanup_fileL st.session_state.uploaded_files after making it [] is ", st.session_state.uploaded_files)
           
# def cleanup_files():
#     print("Inside cleanup_files")
#     uploaded_files = read_uploaded_files()
#     new_uploaded_files = []
#     print("Inside cleanup_files: uploaded_files is ", uploaded_files)
#     for file_info in uploaded_files:
#         print("Inside cleanup_files: file_info is ", file_info)
#         if session_id in file_info['path']:
#             # delete_file(file_info['name'])
#             delete_file(file_info)
#         else:
#             new_uploaded_files.append(file_info)
#     save_uploaded_files(new_uploaded_files)
#     print("Method cleanup_fileL st.session_state.uploaded_files is ", st.session_state.uploaded_files)
#     st.session_state.uploaded_files = []
#     print("Method cleanup_fileL st.session_state.uploaded_files is ", st.session_state.uploaded_files)

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


    # response = requests.delete(f"{DELETE_URL}{file_name}")
    # if response.status_code == 200:
    #     st.success(f"File {file_name} deleted successfully.")
    #     return True
    # else:
    #     st.error(f"Failed to delete file {file_name}.")
    #     st.error(f"Error: {response.status_code}, {response.text}")
    #     return False

# Cleanup files on app rerun
# cleanup_files()

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

# # Ensure session state is initialized
# if "messages" not in st.session_state:
#     st.session_state.messages = []


####################

# def save_file(uploaded_file):
#     """Helper function to save documents to disk via FastAPI."""
#     response = requests.post("http://localhost:8000/uploadfile/", files={"uploaded_file": uploaded_file})
#     if response.status_code == 200:
#         file_info = response.json()
#         st.session_state.uploaded_files.append(file_info)
#         print("File info added to session state:", file_info)
#     else:
#         st.error("File upload failed.")
#         print("File upload failed with status code:", response.status_code)

# def delete_file(file_name):
#     """Helper function to delete documents from disk via FastAPI."""
#     response = requests.delete(f"http://localhost:8000/deletefile/{file_name}")
#     if response.status_code == 200:
#         st.success(f"File {file_name} deleted successfully.")
#         print(f"File {file_name} deleted successfully.")
#     else:
#         st.error(f"Failed to delete file {file_name}.")
#         print(f"Failed to delete file {file_name} with status code:", response.status_code)

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

        # Force a rerun to clear the uploaded file preview
        # st.session_state.file_uploader_key = str(uuid.uuid4())
        # st.experimental_rerun()

        # # Add the new file to the sesssave_upion state and save
        # save_file(uploaded_file)
        # st.success(f"File {file_name} uploaded successfully.")

        # file_path = f"./uploaded_files/{st.session_state.session_id}/{uploaded_file.name}"
        # file_info = {'name': uploaded_file.name, 'path': file_path}
        # # Check if the file already exists in session state
        # if file_info not in st.session_state.uploaded_files:
        #     os.makedirs(os.path.dirname(file_path), exist_ok=True)
        #     with open(file_path, 'wb') as f:
        #         f.write(uploaded_file.getbuffer())
        #     st.session_state.uploaded_files.append(file_info)
        #     # save_uploaded_files(st.session_state.uploaded_files)
        #     save_file(uploaded_file)
        #     st.success(f"File {uploaded_file.name} uploaded successfully.")
        # else:
        #     st.warning(f"File {uploaded_file.name} is already uploaded.")

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


        # Button to remove file
        # if st.button(f"Remove {file_name}"):
        #     #Handle file deletion and session state update
        #     if delete_file(file_name):  # Implement delete_file function
        #         print("after delete file st.session_state.uploaded_files is ", st.session_state.uploaded_files)
        #         st.session_state.uploaded_files.remove(file_info)
        #         print("after delete file st.session_state.uploaded_files is ", st.session_state.uploaded_files)
        #         save_uploaded_files(st.session_state.uploaded_files)  # Implement save_uploaded_files function
        #         st.rerun()  # Refresh the page after deletion

        #     else:
        #         st.error(f"Failed to delete file {file_name}")

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


    #  # Handle file uploads
    # if uploaded_files:
    #     for uploaded_file in uploaded_files:
    #         save_file(uploaded_file)

    # # Display and handle file removals
    # st.write("Uploaded files:")
    # if st.session_state.uploaded_files:
    #     for idx, file_info in enumerate(st.session_state.uploaded_files):
    #         file_name = file_info['filename']
    #         unique_key = f"{file_name}-{idx}-{datetime.now().timestamp()}"
    #         if st.button(f"Remove {file_name}", key=unique_key):
    #             response = delete_file(file_name)
    #             if response.get('status') == 'success':

    #                 delete_file(file_name)
    #                 st.rerun()
    # else:
    #     st.write("No files uploaded yet.")
            
    # # Save the uploaded files and update session state
    # if uploaded_files:
    #     # Clear previous files from the sidebar and delete them via FastAPI
    #     for file_info in st.session_state.uploaded_files:
    #         delete_file(file_info['name'])
    #     st.session_state.uploaded_files = []

    #     for uploaded_file in uploaded_files:
    #         response = save_file(uploaded_file)
    #         if response.get('status') == 'success':
    #             st.session_state.uploaded_files.append({'name': uploaded_file.name, 'data': uploaded_file})



    # Ensure only the name of the last uploaded document remains in the sidebar
    # if len(st.session_state.uploaded_files) > 0:
    #     print("Ensure only the last file is displayed!")
    #     st.write(f"Uploaded file: {st.session_state.uploaded_files[-1]['name']}")

    # for uploaded_file in uploaded_files:
    #     print("Is this printed after the upload?")
    #     save_file(uploaded_file)

# for file_info in st.session_state.uploaded_files:
#     file_path = file_info['name']
#     file_name = os.path.basename(file_path)
#     if st.sidebar.button(f"Remove {file_name}"):
#         # Remove file from session state and filesystem
#         st.session_state.uploaded_files.remove(file_info)
#         delete_file(file_path)
#         # Break to avoid runtime error due to state change during iteration
#         break



# # Accept user input
# if prompt := st.chat_input("Ask me anything!"):
#     st.session_state.messages.append({"role": "user", "content": prompt})

#     with st.chat_message("user"):
#         st.markdown(prompt)

#     with st.chat_message("assistant"):
#         response = requests.post(
#             "http://localhost:8000/query",
#             json={"prompt": prompt, "max_new_tokens": max_new_tokens, "k": k, "temperature": temperature},
#         )
#         if response.status_code == 200:
#             response_data = response.json()
#             answer = response_data["response"]
#             st.markdown(answer)
#             st.session_state.messages.append({"role": "assistant", "content": answer})
#         else:
#             st.write(f"Error: {response.status_code}")
        
#         print("Response content:", response.content)
#     try:
#         answer = response.json().get("response")
#         st.session_state.messages.append({"role": "assistant", "content": answer})
#     except ValueError as e:
#         st.error(f"Error parsing response: {e}")
#         st.write(response.text)

#         print("Response content:", response.content)
#     save_state()

# # Remove uploaded files on app termination
# clear_uploaded_files()

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

    # try:
    #     print("Print the second time the previous response?")
    #     answer = response.json().get("response")
    #     st.session_state.messages.append({"role": "assistant", "content": answer})
    #     # append_message({"role": "assistant", "content": answer})
    # except ValueError as e:
    #     st.error(f"Error parsing response: {e}")
    #     st.write(response.text)

    #     print("Response content:", response.content)
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

########################

# import os
# import streamlit as st
# from model import ChatModel
# import rag_util
# import time


# FILES_DIR = os.path.normpath(
#     os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
# )
# os.makedirs(FILES_DIR, exist_ok=True)
# st.title("🤖 LLM Chatbot RAG Assistant")
# st.markdown('Model: :orange[mistralai/Mistral-7B-Instruct-v0.2]')

# @st.cache_resource
# def load_model():
#     model = ChatModel(model_id="mistralai/Mistral-7B-Instruct-v0.2", device="cuda")
#     return model

# @st.cache_resource
# def load_encoder():
#     encoder = rag_util.Encoder(
#         model_name="sentence-transformers/all-MiniLM-L12-v2", device="cpu"
#     )
#     return encoder

# model = load_model()  # load our models once and then cache it
# encoder = load_encoder()

# def save_file(uploaded_file):
#     """helper function to save documents to disk"""
#     file_path = os.path.join(FILES_DIR, uploaded_file.name)
#     os.makedirs(FILES_DIR, exist_ok=True)
#     with open(file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())
#     return file_path

# with st.sidebar:
#     st.subheader('Parameters:')
#     max_new_tokens = st.number_input("max_new_tokens", 128, 1024, 256,help='input range: 128 ~ 1024')
#     k = st.slider("k: Amount of documents to return", 1, 10, 3,help='input range: 1 ~ 10')
#     temperature = st.number_input("temperature", 0.00,1.00,0.01,help='input range: 0.00 ~ 1.00')
#     st.divider()
#     uploaded_files = st.file_uploader(
#         "Upload PDFs for context", type=["PDF", "pdf"], accept_multiple_files=True
#     )
#     @st.cache_resource
#     def load_database(file_paths):
#         docs = rag_util.load_and_split_pdfs(file_paths)
#         DB = rag_util.PGVectorDb(docs=docs, embedding_function=encoder.embedding_function)
#         return DB
#     file_paths = []
#     upload_progress_text = "Uploading files in progress. Please wait."
#     upload_bar = st.progress(0, text=upload_progress_text)
#     progress_batch = 1 / max(len(uploaded_files),1)
#     for i in range(len(uploaded_files)):
#         file_paths.append(save_file(uploaded_files[i]))
#         upload_bar.progress((i + 1) * progress_batch, text=upload_progress_text)

#     upload_bar.empty()

#     @st.cache_data
#     def check_if_file_change(file_paths):
#         return file_paths
#     file_paths_check = check_if_file_change(file_paths)
#     if uploaded_files != []:
#         if file_paths_check == file_paths:
#             DB =  load_database(file_paths) 
#         else:
#             load_database.clear()
#             check_if_file_change.clear()
#             DB =  load_database(file_paths)
            

#         # with st.status("Embedding Files...", expanded=True) as status:
#         #     st.write("Splitting Files...")
#         #     docs = rag_util.load_and_split_pdfs(file_paths)
#         #     st.write("Load into Database...")
#         #     DB = rag_util.PGVectorDb(docs=docs, embedding_function=encoder.embedding_function)
#         #     status.update(label="Embedding complete!", state="complete", expanded=False)
#         #     return DB
#     else:
#         load_database.clear()
#         # Iterate over all files in the folder
#         for filename in os.listdir(FILES_DIR):
#             file_path = os.path.join(FILES_DIR, filename) 
#             # Check if the path is a file (not a directory)
#             if os.path.isfile(file_path):
#                 # Delete the file
#                 os.remove(file_path)    
        
            

# # Initialize chat history
# if "messages" not in st.session_state:
#     st.session_state.messages = []

# # Display chat messages from history on app rerun
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# memory_reset_trigger = False
# # Accept user input
# if prompt := st.chat_input("Ask me anything!"):
#     # Add user message to chat history
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     # Display user message in chat message container
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     # Display assistant response in chat message container
#     with st.chat_message("assistant"):
#         user_prompt = st.session_state.messages[-1]["content"]
#         context = (
#             None if uploaded_files == [] else DB.similarity_search(user_prompt, k=k)
#         )
#         database=(
#             None if uploaded_files == [] else DB.get_db()
#         )
#         with st.spinner('Generating answer...'):
#             answer = model.generate(
#                 user_prompt, 
#                 db = database,
#                 context=context, 
#                 max_new_tokens=max_new_tokens, 
#                 temperature=temperature, 
#                 k=k,
#         )
#         response = st.write(answer)
#     st.session_state.messages.append({"role": "assistant", "content": answer})

# if st.button("Reset Conversation"):
#     memory_reset_trigger = True
#     model.check_memory(memory_reset_trigger=memory_reset_trigger)
#     st.session_state.messages=[]
    
