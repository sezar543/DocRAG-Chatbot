import os
from pathlib import Path
from model import ChatModel
from rag_util import Encoder  # Replace 'your_module' with the actual module name where ChatModel and Encoder are defined

CACHE_DIR = "/home/ubuntu/uae-chatbot/erming/streamlit/models"

def list_cached_files(cache_dir):
    for path in Path(cache_dir).rglob('*'):
        if path.is_file():
            print(path)

if __name__ == "__main__":
    # List files before loading models
    print("Cached files before loading models:")
    list_cached_files(CACHE_DIR)

    # Initialize models
    chat_model = ChatModel()
    encoder = Encoder()

    # List files after loading models
    print("Cached files after loading models:")
    list_cached_files(CACHE_DIR)
