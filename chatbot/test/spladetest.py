import base64
import os
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import torch
import json
from dotenv import load_dotenv
# Import embedding models
from langchain_community.embeddings import HuggingFaceEmbeddings
from pinecone_text.sparse import SpladeEncoder

# Import embedding models
from transformers import AutoModelForMaskedLM, AutoTokenizer



# Load JSON data
with open('/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/completed_data/00_harvard_CS50.json', 'r') as file:
    datas = json.load(file)
    
    print(f"DATAS : {datas}")

from pinecone_text.sparse import BM25Encoder

# Make sure to set your PINECONE_API_KEY in your environment variables
load_dotenv()
api_key = os.getenv('PINECONE_API_KEY')
if not api_key:
    raise ValueError("Please set the PINECONE_API_KEY environment variable.")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)



from langchain_community.embeddings import HuggingFaceEmbeddings

# dense embedding model : intfloat/e5-large-v2
model_name = "intfloat/e5-large-v2"
model_kwargs = {'device': 'cpu'}       
encode_kwargs = {'normalize_embeddings': True}
hf = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)


from transformers import AutoTokenizer
from collections import Counter

# Define the vector dimension based on your model's output (e.g., 384 for 'all-MiniLM-L6-v2')
vector_dimension = 1024

# The name of your Pinecone index
index_name = 'curriculum'.lower()  # Ensure the name is lowercase

# Create an index if it does not exist
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=vector_dimension,
        metric='dotproduct', 
        spec=ServerlessSpec(
            cloud='gcp',
            region='us-central1'
        )
    )

# Extract the host if it's available
index_host = None
try:
    index_description = pc.describe_index(name=index_name)
    print(f"\nindex_description : \n{index_description}")
    index_host = index_description['host'] 
    print(f"\n index host : \n{index_host}\n")
except Exception as e:
    print(f"An error occurred: {e}")

if index_host is None:
    raise Exception("Unable to retrieve index 'host' information.")

# Initialize Pinecone Index with the host
index = pc.Index(name=index_name, host=index_host)

def encode_id(doc_id):
    # Convert the document ID to bytes and encode it into Base64 to ensure it's ASCII
    encoded_bytes = base64.b64encode(doc_id.encode('utf-8'))
    # Decode the Base64 bytes into a string
    return encoded_bytes.decode('utf-8')


# Function to upload documents to Pinecone
def encode_and_index_documents(datas):
    upserts = []
    if 'data' in datas:
        for doc in datas['data']:
            print(f"\n DOCUMENT : \n {doc}\n")
            try:
                # Embed using HuggingFace model for dense embeddings
                dense_embedding = hf.embed_documents([doc['text']])[0]  # Use 'context' instead of 'text'
                print("\nHUGGING FACE FINISHED\n")
                # Ensure the embedding is a list of floats
                if isinstance(dense_embedding, np.ndarray):
                    dense_embedding = dense_embedding.tolist()
                elif isinstance(dense_embedding, torch.Tensor):
                    dense_embedding = dense_embedding.cpu().numpy().tolist()

                summarization = doc['summarization']
                print(f"summarization : {summarization}")
                
                # Initialize Splade
                splade = SpladeEncoder()
                sparse = splade.encode_documents(summarization)

                # Print or process the sorted dictionary as needed
                print(sparse)
                print(f"indices : {sparse['indices']}")

                # Prepare sparse_values in the correct format for Pinecone
                sparse_values = {
                    "indices": sparse['indices'],
                    "values": sparse['values']
                }
                
                # Encode document ID
                encoded_id = doc['id']
                metadata = {"text": doc["text"], "summarization": doc["summarization"],"timestamp":doc["timestamp"]}  # Metadata should be a dictionary
                # metadata = {"text": doc["text"]}  # no timestamp
                # Prepare upsert data structure
                upsert = {
                    'id': encoded_id,  # Now this is a string, as required
                    'values': dense_embedding,
                    'sparse_values': sparse_values,
                    "metadata": metadata
                }
                upserts.append(upsert)

            except Exception as e:
                print(f"Failed to embed document {doc['id']}: {e}")
                continue

    # Upsert documents to Pinecone in batches
    if upserts:
        response = index.upsert(vectors=upserts)
        print(response)  # Check the response from Pinecone


# Call the function with your documents
encode_and_index_documents(datas)
