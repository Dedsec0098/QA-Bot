import json
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated import
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

# Step 3: Load and preprocess JSONL data
print("Loading JSONL data...")

# Load documents from JSONL file
docs = []
with open('contributor_questions.jsonl', 'r') as file:
    for line in file:
        if line.strip():  # Skip empty lines
            data = json.loads(line)
            # Create a document from each Q&A pair
            # Updated to use the actual field names in your JSONL file
            if 'question' in data and 'answer' in data:
                docs.append(
                    Document(
                        page_content=f"Question: {data['question']}", 
                        metadata={'id': data['id'], 'answer': data['answer']}
                    )
                )

print(f"Loaded {len(docs)} documents")

# Check if we have documents before proceeding
if not docs:
    print("ERROR: No documents were loaded. Please check your JSONL file format.")
    exit(1)

# Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len
)
splits = text_splitter.split_documents(docs)
print(f"Split into {len(splits)} chunks")

# Step 4: Create Vector Database
print("Creating embeddings and vector store...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cpu'}
)

# Create vector store
vectorstore = FAISS.from_documents(
    documents=splits,
    embedding=embeddings
)

# Save vector store
print("Saving vector store to disk...")
vectorstore.save_local("faiss_index")
print("Vector store saved successfully! You can now use it for retrieval.")