

def TokenizerProps(tokenizer):
    print("\n=== Core Props ===")
    print(f"Tokenizer class: {type(tokenizer).__name__}")
    print(f"Vocab size: {tokenizer.vocab_size}")          
    print(f"Model max length: {tokenizer.model_max_length}")  
    print(f"Padding side: {tokenizer.padding_side}") 
    print("\n=== Special Tokens ===")
    print(f"CLS token: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
    print(f"SEP token: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
    print(f"PAD token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"UNK token: {tokenizer.unk_token} (ID: {tokenizer.unk_token_id})")
    print(f"BOS token: {tokenizer.bos_token} (ID: {tokenizer.bos_token_id})")
    print(f"EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")
    print("\n=== Tokenizer Config ===")
    try:
        print(tokenizer.config)  # Full tokenizer configuration
    except:
        print("no config detected")
    
    text = "Hello, how are you?"
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.encode(text)

    print("\n=== Tokenization Example ===")
    print(f"Text: '{text}'")
    print(f"Tokens: {tokens}")  # e.g., ['Hello', ',', ' how', ' are', ' you', '?']
    print(f"Token IDs: {ids}")   # e.g., [1, 15043, 1111, 1678, 1122, 136, 2]

    print("\n=== Advanced Properties ===")
    print(f"Added tokens: {len(tokenizer.get_added_vocab())}")  # Custom-added tokens
    print(f"Is fast tokenizer? {tokenizer.is_fast}")  # True for Rust-based tokenizers
    try:
        print(f"Language: {tokenizer.language}")          # e.g., 'en'
    except:
        print("No language deetected")          # e.g., 'en'
        
def ModelProps(model):
    print("=== Basic Model Info ===")
    print(f"Model class: {type(model).__name__}")
    print(f"Model path: {model.name_or_path}")
    print(f"Device: {model.device}")
    print(f"Data type: {model.dtype}")
    
    print("\n=== Architecture ===")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Layers: {model.config.num_hidden_layers}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Attention heads: {model.config.num_attention_heads}")
    print(f"Vocab size: {model.config.vocab_size}")
    
    print("\n=== Config Summary ===")
    print({k: v for k, v in model.config.to_dict().items() 
           if not k.startswith('_') and not isinstance(v, (dict, list))})
    
    if hasattr(model, 'generation_config'):
        print("\n=== Generation Config ===")
        print(model.generation_config)

def ChatbotProps(chatbot):
    try:
        print("=== Pipeline Properties ===")
        print(f"Task: {getattr(chatbot, 'task', 'N/A')}")
        print(f"Device: {getattr(chatbot, 'device', 'N/A')}")
        
        print("\n=== Model ===")
        if hasattr(chatbot, 'model'):
            print(f"Model: {chatbot.model.__class__.__name__}")
            print(f"Device: next(chatbot.model.parameters()).device")
        else:
            print("Model not loaded!")
            
        print("\n=== Tokenizer ===")
        if hasattr(chatbot, 'tokenizer'):
            print(f"Tokenizer: {chatbot.tokenizer.__class__.__name__}")
        else:
            print("Tokenizer missing!")
            
    except Exception as e:
        print(f"Error inspecting pipeline: {str(e)}")


    
