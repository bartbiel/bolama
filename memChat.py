from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path
from props import TokenizerProps, ModelProps, ChatbotProps
from pydantic import BaseModel

from langchain.prompts import PromptTemplate
from langchain.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from langchain.chains import LLMChain


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
#TokenizerProps(tokenizer)


model = AutoModelForCausalLM.from_pretrained(
    Mistral_snapshot,
    local_files_only=True,
    torch_dtype=torch.float16,  
    device_map="cpu",
    low_cpu_mem_usage=True
)
#ModelProps(model)

# Create LangChain pipeline
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    #device="cpu",  # or "cuda" if you have GPU
    max_new_tokens=30,
    temperature=0.7,
    do_sample=True,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id  # Important for some models
)

# 3. Convert to LangChain pipeline
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Now you can use this in your LangChain setup
memory = ConversationBufferMemory(memory_key="chat_history")

template = """You are a helpful AI assistant that provides extremely concise answers. 
Strictly follow these rules:
1. Your response must be exactly 2 sentences
2. Be direct and to the point
3. Use the following context only if relevant

Previous conversation: {chat_history}
New question: {question}

Response (exactly 2 sentences):"""
prompt = PromptTemplate.from_template(template)

conversation_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
    verbose=True
)
class ChatRequest(BaseModel):
    user_input: str  # Must match React's payload key
    conversation_id: str = None  # Optional for tracking conversations

@app.post("/chat")
async def generate_text(request: ChatRequest):
    try:
       response = conversation_chain.run(question=request.user_input)
       return {"response": response}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

print ("===memchat finished===")