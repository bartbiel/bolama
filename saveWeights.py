from transformers import AutoTokenizer, AutoModelForCausalLM
import grants
from pathlib import Path
from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


model_name = "mistralai/Mistral-7B-Instruct-v0.3"
local_path = Path(grants.Mistral_PATH)
hf_token=grants.API_HF

print(f"=======================saveWeights:{current_time}==============")
model = AutoModelForCausalLM.from_pretrained(model_name, token=hf_token)
model.save_pretrained(local_path)  # Save locally

#tokenizer = AutoTokenizer.from_pretrained(local_path, token=hf_token)
#model = AutoModelForCausalLM.from_pretrained(local_path,  token=hf_token)
#tokenizer.save_pretrained(local_path)
#model.save_pretrained(local_path)
