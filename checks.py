from pathlib import Path
import grants

def checkMistral(model_dir: str):
    model_dir = Path(model_dir)
    #model_dir = Path.home() / "mistral_models" / "7B-Instruct-v0.3"

    # Check if the directory exists
    if model_dir.exists():
        print(f"Files in {model_dir}:")
        for file in model_dir.iterdir():
            print(" -", file.name)
    else:
        print(f"The directory {model_dir} does not exist.")

checkMistral(grants.Mistral_PATH)