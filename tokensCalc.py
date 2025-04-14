from transformers import AutoTokenizer
from pathlib import Path
import grants

# Path to model and text file
model_path = Path(grants.Mistral_PATH)
text_file_path = Path("./resources/res1.txt") 

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Read the text from the file
with open(text_file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize the text
tokens = tokenizer.encode(text, return_tensors=None)

# Calculate counts
num_tokens = len(tokens)
num_words = len(text.split())
num_characters = len(text)

# Print results
print(f"Number of tokens: {num_tokens}")
print(f"Number of words: {num_words}")
print(f"Number of characters: {num_characters}")


# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

# Ensure pad token is set
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Read the text from the file
with open(text_file_path, "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize the text
tokens = tokenizer.encode(text, return_tensors=None)



