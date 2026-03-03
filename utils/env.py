from dotenv import load_dotenv
import os

def setup_env():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if hf_token:
        os.environ["HF_TOKEN"] = hf_token