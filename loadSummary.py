from summaryBcknd import generate_summary, load_text_from_file
input_file = "./resources/res1.txt"
    
# Load text from file
text = load_text_from_file(input_file)
if not text:
    print("Failed to load text from file.")
    
    
print("\n=== Original Text ===")
print(f"Length: {len(text)} characters")
print("---------------------")
print(text[:500] + "..." if len(text) > 500 else text)  # Print first 500 chars
    
# Generate summary
max_length = 300  # Adjust summary length as needed
summary = generate_summary(text, max_length)
    
if summary:
    print("\n=== Generated Summary ===")
    print(f"Length: {len(summary)} characters (max: {max_length})")
    print("---------------------")
    print(summary)
else:
    print("Failed to generate summary.")