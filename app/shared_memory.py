# shared_memory.py
from langchain.memory import ConversationBufferWindowMemory

memory_store = {}

def get_memory(user_id: str):
    if user_id not in memory_store:
        memory_store[user_id] = ConversationBufferWindowMemory(k=3)
    return memory_store[user_id]

def reset_memory(user_id: str):
    if user_id in memory_store:
        memory_store[user_id].clear()
        print(f'Memory for user {user_id} has been cleared!')