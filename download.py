from huggingface_hub import snapshot_download
from huggingface_hub import login
from pathlib import Path
import grants
login(grants.API_HF)

mistral_models_path = Path(grants.MistralDIR).joinpath('mistral_models', '7B-Instruct-v0.3')
mistral_models_path.mkdir(parents=True, exist_ok=True)

snapshot_download(repo_id="mistralai/Mistral-7B-Instruct-v0.3", allow_patterns=["params.json", "consolidated.safetensors", "tokenizer.model.v3"], local_dir=mistral_models_path)
