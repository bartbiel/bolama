from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from pathlib import Path
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from datetime import datetime
import grants

# ==================== Time & Setup ====================
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(f"=======================Text Summary Generator: {current_time} =======================")

# ==================== Load Model and Tokenizer ====================
model_path = Path(grants.Mistral_PATH)
Mistral_snapshot = Path(grants.Mistral_snapshot)
print(f"Using model from: {model_path}")

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    print("Configured padding token using EOS token")

model = AutoModelForCausalLM.from_pretrained(
    Mistral_snapshot,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True,
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.eos_token_id,
    truncation=True
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

# ==================== Load PDF and Text Documents ====================
pdf_loader = PyPDFLoader("./resources/co.pdf")
text_loader = TextLoader("./resources/res1.txt", encoding="utf-8")

pdf_docs = pdf_loader.load()
text_docs = text_loader.load()
all_docs = pdf_docs + text_docs

# ==================== Split Documents into Chunks ====================
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(all_docs)

# ==================== Embed and Index ====================
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(docs, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ==================== Create RAG Chain ====================
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# ==================== Ask a Question ====================
query = "Find expert in the loop role"
response = rag_chain({"query": query})

print("====== RAG Answer ======")
print(response["result"])

print("\n====== Source Documents Used ======")
for doc in response["source_documents"]:
    print(f"- {doc.metadata.get('source', 'Unknown')} | Excerpt: {doc.page_content[:200]}...\n")
