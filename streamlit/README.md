# Streamlit APP Setting Instruction
## Files setting before building Docker image
### `.env`
HUGGINGFACEHUB_API_TOKEN = <your_huggingface_token> \
POSTGRES_PASS = <your_postgres_password>
### `rag_util.py` 
In Class `PGVectorDb.__init__`, change the following parameter for postgres sql connection: \
`postgres_user = <your_postgres_user>` \
`postgres_password = os.getenv("POSTGRES_PASS") # Do not need to change` \
`postgres_host = 'localhost:5432' # Do not need to change` \
`postgres_database = <your_postgres_database>` 

### `requirements.txt` *
If GPU use following link: \
`--extra-index-url https://download.pytorch.org/whl/cu118` \
If CPU only change to following link: \
`--extra-index-url https://download.pytorch.org/whl/cpu` \
Comment out the one that Not used.
### `model.py` *
If CPU only, comment out the following in Class `ChatModel.__init__`: \
`quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` \
\
and for `self.model = AutoModelForCausalLM.from_pretrained()` function: \
`quantization_config=quantization_config,` \
### `streamlit_app.py` **
If the model and embedding structures are same, simply change the `model_id` and `model_name` kwargs should be able to change the models
```
@st.cache_resource
def load_model():
    model = ChatModel(model_id="meta-llama/Meta-Llama-3-8B-Instruct", device="cuda")
    return model

@st.cache_resource
def load_encoder():
    encoder = rag_util.Encoder(
        model_name="sentence-transformers/all-mpnet-base-v2", device="cpu"
    )
    return encoder
```
\
__\* essential whether GPU or only CPU available__ \
__\** optional and might need further change__ 
## Docker
### Container Structure
```
/
├── app  # This is the container WORKDIR
│   ├── Dockerfile 
│   ├── .env
│   ├── model.py
│   ├── rag_util.py
│   ├── streamlit_app.py
│   └── requirements.txt
├── models  # This saves the cache of the language model and embedding model
│   ├── some_language_model
│   └── some_embedding_model
└── files  # This saves the file which user upload from the streamlit app
    └── some pdf files
```
### Build Docker image
1. `cd` to `app/` dir
2. `docker build -t streamlit .`
3. `docker images` to check if the image created successfully
### Run Docker container
in order to map the host postgres sql port to container port, need to make the network setting in container as 'host'. \
`docker run --network host streamlit`
## Streamlit
Streamlit is on port '8501', since the network of container is set to 'host', the app can be reach in browser with \
`localhost:8501`
## Disk space usage
- Docker image: 3.2gb (cpu only),or 10gb(with cuda) 
- Language model: 15gb (meta-llama--Meta-Llama-3-8B-Instruct) 
- Embedding model: 0.4gb (sentence-transformers--all-mpnet-base-v2)

## COMMENT
Model Quantization into 8 bit or 4 bit not working for CPU only interface, need further invenstigation. \
Model can initialize with CPU only, but takes very long time to answer a question, and during the session, a lot of memory and disk space are used. (tested with `AMD Ryzen 9 5980HS`, CPU cores: 8, with around 14gb free memory) \

