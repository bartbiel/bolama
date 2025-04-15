import faiss
import pickle
from langchain.docstore.in_memory import InMemoryDocstore  # Import if needed

# Load the metadata tuple
with open("./faiss_index/index.pkl", "rb") as f:
    metadata = pickle.load(f)

# Inspect the tuple structure
print("Metadata type:", type(metadata))  # Should output: <class 'tuple'>
print("Tuple length:", len(metadata))  # Should output: 2

# Check the types of each tuple element
element_types = [type(item) for item in metadata]
print("Element types:", element_types)  # e.g., [<class 'InMemoryDocstore'>, <class 'dict'>]

# Handle InMemoryDocstore (first element)
if isinstance(metadata[0], InMemoryDocstore):
    print("\nInMemoryDocstore contents:")
    # List all document IDs in the docstore
    doc_ids = list(metadata[0]._dict.keys())  # Access internal dictionary
    print("Number of documents:", len(doc_ids))
    print("Sample document ID:", doc_ids[0])
    # Access a sample document's content
    sample_doc = metadata[0]._dict[doc_ids[0]]
    print("Sample document content:", sample_doc.page_content[:100] + "...")  # Preview first 100 chars

# Handle second element (usually a dict or list)
if len(metadata) > 1:
    print("\nSecond element type:", type(metadata[1]))
    if isinstance(metadata[1], dict):
        print("Sample value:", next(iter(metadata[1].values())))
    elif isinstance(metadata[1], list):
        print("List length:", len(metadata[1]))
        print("Sample item:", metadata[1][0])

# FAISS Index Info (unchanged)
index = faiss.read_index("./faiss_index/index.faiss")
print("\nFAISS Index Info:")
print("Vector dimension (d):", index.d)
print("Number of vectors:", index.ntotal)
print("Index type:", type(index))