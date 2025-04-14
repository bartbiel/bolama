from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import grants
from datetime import datetime

current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"=======================Text Summary Generator:{current_time}==============")

# Load Mistral 7B locally
model_path = Path(grants.Mistral_PATH)
Mistral_snapshot = Path(grants.Mistral_snapshot)
print(f"Using model from: {model_path}")

# Load tokenizer and model from local path
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

# Configure tokenizer with padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Use EOS token as padding token
    print("Configured padding token using EOS token")

model = AutoModelForCausalLM.from_pretrained(
    Mistral_snapshot,
    local_files_only=True,
    torch_dtype=torch.float16,  
    device_map="auto",  # Changed to auto for better device handling
    low_cpu_mem_usage=True
)

# Create text generation pipeline with proper configuration
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,  # Explicitly set maximum new tokens
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,  # Use configured padding token
    eos_token_id=tokenizer.eos_token_id,
    truncation=True  # Enable truncation to model's max length
)

# Wrap the pipeline using the updated HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# Define summarization prompt template
summarization_prompt = """
Please provide a concise summary of the following text. Focus on the key points and main ideas.
Keep the summary under {max_length} characters.

Text:
{text}

Summary:
"""

prompt = PromptTemplate(
    input_variables=["text", "max_length"],
    template=summarization_prompt
)

# Create the summarization chain using the new Runnable approach
summarize_chain = (
    {"text": RunnablePassthrough(), "max_length": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

def load_text_from_file(file_path):
    """Load text content from a file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def generate_summary(text, max_length=512):
    """Generate summary for the given text"""
    try:
        # Truncate input text if it's too long for the model
        max_input_length = tokenizer.model_max_length - 100  # Leave room for prompt
        if len(text) > max_input_length:
            print(f"Warning: Truncating input text from {len(text)} to {max_input_length} characters")
            text = text[:max_input_length]
        
        summary = summarize_chain.invoke({
            "text": text,
            "max_length": min(max_length, 512)  # Ensure we don't exceed model limits
        })
        
        # Clean up the output
        summary = summary.replace(text, "").strip()
        return summary
    except Exception as e:
        print(f"Error generating summary: {e}")
        return None