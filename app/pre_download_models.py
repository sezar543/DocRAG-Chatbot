import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


ACCESS_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")

model_encoder = os.getenv("model_encoder")
LLM_model_id = os.getenv("LLM_model_id")

print("ACCESS_TOKEN ", ACCESS_TOKEN)

def download_and_cache_models():
    # Set cache directory
    CACHE_DIR = "/home/ubuntu/uae-chatbot/erming/streamlit/models"

    # Model details
    model_id = LLM_model_id
    # encoder_model_name = "sentence-transformers/all-MiniLM-L12-v2"
    encoder_model_name = model_encoder
    
    # Download and cache the chat model
    print("Downloading and caching chat model...")
    AutoTokenizer.from_pretrained(model_id=model_id, cache_dir=CACHE_DIR, token=ACCESS_TOKEN)
    AutoModelForCausalLM.from_pretrained(
        model_id,
        cache_dir=CACHE_DIR,
        token=ACCESS_TOKEN,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    )

    # Download and cache the encoder model
    print("Downloading and caching encoder model...")
    SentenceTransformer(model_name_or_path=encoder_model_name, cache_folder=CACHE_DIR)

    print("Models downloaded and cached successfully.")

if __name__ == "__main__":
    download_and_cache_models()
