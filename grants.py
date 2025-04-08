
from dotenv import load_dotenv
from pathlib import Path
load_dotenv()
import os
API_HF=os.getenv("API_HF")
VENF_PATH=os.getenv("venv_PATH")
#Mistral_PATH=Path.home() / "mistral_models" / "7B-Instruct-v0.3"
Mistral_PATH=os.getenv("Mistral_PATH")
Mistral_Weigths_PATH=os.getenv("Mistral_Weigths_PATH")
MistralDIR=os.getenv("MistralDIR")
Mistral_snapshot=os.getenv("Mistral_snapshot")



