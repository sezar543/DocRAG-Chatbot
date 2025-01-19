import os

def list_cached_files(cache_dir):
    for root, dirs, files in os.walk(cache_dir):
        for file in files:
            print(os.path.join(root, file))

list_cached_files("/home/ubuntu/uae-chatbot/erming/streamlit/models")