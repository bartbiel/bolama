from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import grants

# Set up the model directory
model_dir = grants.Mistral_PATH

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)

# Function to interact with the model
def chat_with_mistral(user_input):
    # Encode the input text
    inputs = tokenizer(user_input, return_tensors="pt")
    
    # Generate a response from the model
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'], 
            max_length=256,  # Maximum tokens to generate
            temperature=0.7,  # Randomness
            top_p=0.9,  # Nucleus sampling
            eos_token_id=tokenizer.eos_token_id
        )
    
    # Decode and print the model's response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Example interaction
user_input = "Explain the concept of Machine Learning."
response = chat_with_mistral(user_input)
print("Mistral's response:", response)
