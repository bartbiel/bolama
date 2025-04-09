from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path
from props import TokenizerProps, ModelProps, ChatbotProps
from pydantic import BaseModel

import torch
import grants

from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"=======================webChat:{current_time}==============")
app = FastAPI()

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Mistral 7B locally
model_path = Path(grants.Mistral_PATH)
Mistral_snapshot=Path(grants.Mistral_snapshot)
print(f"Using model from: {model_path}")
print(f"model path for webChat= {model_path}")
# Load tokenizer and model from local path (no need for authentication token)
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True  # Forces it to load locally, not from Hugging Face Hub
)
TokenizerProps(tokenizer)


model = AutoModelForCausalLM.from_pretrained(
    Mistral_snapshot,
    local_files_only=True,
    torch_dtype=torch.float16,  
    device_map="cpu",
    low_cpu_mem_usage=True
)
ModelProps(model)

chatbot = pipeline("text-generation", model=model, tokenizer=tokenizer)
ChatbotProps(chatbot)


class ChatRequest(BaseModel):
    user_input: str  # Must match React's payload key

@app.post("/chat")
async def generate_text(request: ChatRequest):
    try:
        response = chatbot(
            request.user_input,
            max_length=100,
            do_sample=True,
            temperature=0.7,
        )
        return {"response": response[0]["generated_text"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

print ("=========================end========================================")