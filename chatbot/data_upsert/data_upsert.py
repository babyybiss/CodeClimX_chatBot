import base64
import os
import numpy as np
from pinecone import Pinecone, ServerlessSpec
import torch
import json
from dotenv import load_dotenv
from transformers import AutoTokenizer
from splade.models.transformer_rep import Splade
from sentence_transformers import SentenceTransformer

#dense model
device = 'cpu'
dense_model = SentenceTransformer(
    'msmarco-bert-base-dot-v5',
    device=device
)

# sparse model
sparse_model_id = 'naver/splade-cocondenser-ensembledistil'
sparse_model = Splade(sparse_model_id, agg='max')
sparse_model.to('cpu')
sparse_model.eval()
tokenizer = AutoTokenizer.from_pretrained(sparse_model_id)


load_dotenv()

api_key = os.getenv('PINECONE_API_KEY')
if not api_key:
    raise ValueError("Please set the PINECONE_API_KEY environment variable.")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

# Load JSON data
with open('/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/completed_data/00_harvard.json', 'r') as file:
    datas = json.load(file)
    
    print(f"DATAS : {datas}")


# Define the vector dimension based on your model's output (e.g., 384 for 'all-MiniLM-L6-v2')
vector_dimension = 768

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

    for doc in datas.get('data', []):
        text = doc['text']
        
        ## dense embedding
        dense_vecs = dense_model.encode(text)
        print("\n\nSHAPE  ", dense_vecs.shape)
        dim = dense_model.get_sentence_embedding_dimension()
        print("\n\nDIMENSION  ", dim)
        
        ## sparse embedding
        # Transform the document text
        tokens = tokenizer(text, return_tensors='pt')
        
        with torch.no_grad():
            sparse_emb = sparse_model(
                d_kwargs=tokens.to(device)
            )['d_rep'].squeeze()
        sparse_emb.shape
        
        indices = sparse_emb.nonzero().squeeze().cpu().tolist()
        values = sparse_emb[indices].cpu().tolist()
        sparse = {'indices': indices, 'values': values}
        
        print(f"\n\n SPARSE_VECTOR : {sparse}")

        idx2token = {idx: token for token, idx in tokenizer.get_vocab().items()}
        sparse_dict_tokens = {
            idx2token[idx]: round(weight, 2) for idx, weight in zip(indices, values)
        }
        
        
        # sort so we can see most relevant tokens first (visulaization)
        sparse_dict_tokens = {
            k: v for k, v in sorted(
                sparse_dict_tokens.items(),
                key=lambda item: item[1],
                reverse=True
            )
        }
        print(sparse_dict_tokens)
        
        # prepare upsert object
        encoded_id = doc['id']
        metadata = {"text": doc["text"]}
        upsert = {
            'id': encoded_id,
            'values': dense_vecs.tolist(),
            'sparse_values': sparse,
            "metadata": metadata
        }
        upserts.append(upsert)

    # Upsert documents to Pinecone in batches
    if upserts:
        response = index.upsert(vectors=upserts)
        print(response)  # Check the response from Pinecone


# Call the function with your documents
encode_and_index_documents(datas)

