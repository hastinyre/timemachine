import os
import pinecone
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.schema import TransformComponent # Import TransformComponent from schema
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.pinecone import PineconeVectorStore
from dotenv import load_dotenv
from typing import List, Sequence # Keep Sequence import

# Load our secret API keys from the .env file
load_dotenv()

# --- 1. Load Our Documents ---
print("Loading documents from ./data folder...")
reader = SimpleDirectoryReader("./data")
documents = reader.load_data()
print(f"Loaded {len(documents)} document(s).")

# --- 2. Define the "Author" Tagging Transformation ---
# We create a class that inherits from TransformComponent
# This is the standard way to add custom logic to the pipeline now.
class AddAuthorMetadata(TransformComponent):
    # Specify input and output types for clarity (optional but good practice)
    def __call__(self, nodes: List[Document], **kwargs) -> List[Document]:
        for node in nodes:
            # Get file_name safely
            metadata = node.metadata or {}
            file_name = metadata.get("file_name", "")

            if "The_Prince" in file_name:
                node.metadata["author"] = "Machiavelli"
                node.metadata["work"] = "The Prince"
            elif "Discourses_on_Livy" in file_name:
                node.metadata["author"] = "Machiavelli"
                node.metadata["work"] = "Discourses on Livy"
            elif "The_Art_of_War" in file_name:
                node.metadata["author"] = "Machiavelli"
                node.metadata["work"] = "The Art of War"
            # We can add 'elif' for Aristotle's files later!

        return nodes

# --- 3. Set up the Pinecone "Smart Library" ---
print("Connecting to Pinecone...")
# Initialize Pinecone connection using the API key from .env
# Make sure PINECONE_API_KEY is set in your .env file
try:
    pc = pinecone.Pinecone(api_key=os.environ["PINECONE_API_KEY"])
except Exception as e:
    print(f"Error initializing Pinecone: {e}")
    print("Ensure PINECONE_API_KEY is correctly set in your .env file.")
    exit()

# Get the index object
INDEX_NAME = "pantheon-index" # Ensure this matches the index name you created
print(f"Checking if index '{INDEX_NAME}' exists...")
try:
    # Use list_indexes() which returns an IndexList object
    index_list = pc.list_indexes() 
    
    # Extract the names from the Index objects within the list
    available_indexes = [index.name for index in index_list] 

    if INDEX_NAME not in available_indexes:
        print(f"Index '{INDEX_NAME}' does not exist in the list: {available_indexes}")
        print("Please create the index in the Pinecone console with 1536 dimensions and cosine metric.")
        exit() # Stop if index doesn't exist
    else:
         print(f"Index '{INDEX_NAME}' found in list: {available_indexes}")
except Exception as e:
     # Catch potential API key errors or network issues here
     print(f"Error listing Pinecone indexes: {e}")
     print("Please double-check your Pinecone API key in the .env file and your internet connection.")
     exit()


pinecone_index = pc.Index(INDEX_NAME)

# Tell LlamaIndex how to connect to our specific Pinecone index
vector_store = PineconeVectorStore(pinecone_index=pinecone_index)

print("Pinecone connection and index check successful.")

# --- 4. Define the Full "Ingestion Pipeline" ---
# An IngestionPipeline is a series of steps LlamaIndex will run.
print("Setting up ingestion pipeline...")
try:
    pipeline = IngestionPipeline(
        transformations=[
            # Step 1: Split the documents into small, smart chunks
            SentenceSplitter(chunk_size=1024, chunk_overlap=100),

            # Step 2: Use our custom transformation class
            AddAuthorMetadata(), # Use the class instance here

            # Step 3: Use the OpenAI "Librarian" AI to create vectors
            # Make sure OPENAI_API_KEY is set in your .env file
            OpenAIEmbedding(model="text-embedding-3-small", api_key=os.environ.get("OPENAI_API_KEY")),
        ],
        vector_store=vector_store,  # Step 4: Save the final vectors to Pinecone
    )
except Exception as e:
    print(f"Error setting up pipeline: {e}")
    print("Ensure OPENAI_API_KEY is correctly set in your .env file.")
    exit()

print("Starting ingestion pipeline... This may take a few minutes depending on document size.")

# --- 5. Run the Pipeline! ---
try:
    # Process documents through the pipeline
    pipeline.run(documents=documents, show_progress=True) # Added show_progress
except Exception as e:
    print(f"Error during pipeline execution: {e}")
    print("Check your API keys (OpenAI, Pinecone) and network connection.")
    exit()

print("\n---")
print("All documents have been ingested into Pinecone.")
print(f"Your 'Digital Brain' for Machiavelli is now ready!")
print("---")