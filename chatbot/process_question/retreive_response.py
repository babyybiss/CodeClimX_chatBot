import os
import base64
import torch
from pinecone import Pinecone, ServerlessSpec
from transformers import AutoModelForMaskedLM, AutoTokenizer
from langchain_pinecone import PineconeVectorStore
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from splade.models.transformer_rep import Splade
from pinecone_text.sparse import BM25Encoder
from dotenv import load_dotenv
from konlpy.tag import Okt
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

def tokenizer(text):
    tokens = okt.morphs(text)
    print(f"\n\nTOKENS : {tokens}")
    return tokens


#dense model
from sentence_transformers import SentenceTransformer
device = 'cpu'
dense_model = SentenceTransformer(
    'distiluse-base-multilingual-cased-v1',
    device=device
)


okt = Okt()
load_dotenv()

os.environ['PINECONE_ENVIRONMENT'] = "us-central1"
PINECONE_INDEX_NAME = "curriculum"

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Define the vector dimension based on your model's output (e.g., 384 for 'all-MiniLM-L6-v2')
vector_dimension = 512
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

# Load dense model
model_name = "intfloat/e5-large-v2"
model_kwargs = {'device': 'cpu'}
myEmbedding = {'normalize_embeddings': True}
hf_embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=myEmbedding
)

sparse_model_id = 'naver/splade-cocondenser-ensembledistil'
#from splade.models import Splade
sparse_model = Splade(sparse_model_id, agg='max')
sparse_model.to('cpu')
sparse_model.eval()

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)


namespace = "default"
myPineconeVStore = PineconeVectorStore(
    embedding=hf_embeddings,
    index_name=PINECONE_INDEX_NAME,
    namespace=namespace,
    # ... include other parameters as required by PineconeVectorStore ...
)

vectorIndex = VectorStoreIndexWrapper(vectorstore=myPineconeVStore)

def decode_id(encoded_id):
    try:
        # Attempt to decode the Base64 data and then decode the bytes as UTF-8
        return base64.b64decode(encoded_id).decode('utf-8')
    except UnicodeDecodeError as e:
        # Log the error and the problematic ID
        print(f"Error decoding ID: {encoded_id}")
        print(f"UnicodeDecodeError: {e}")
        return None  # or handle the error as appropriate
    
def hybrid_scale(dense, sparse, alpha: float):
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # scale sparse and dense vectors to create hybrid search vecs
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def hybrid_search(query, top_k=4, alpha=float):
    # Encode the query using the dense model
    #dense_embedding = hf_embeddings.embed_query(query)
    d_embedding = dense_model.encode(query)
    
    # Transform the document text
    tokenizer = AutoTokenizer.from_pretrained(sparse_model_id)
    tokens = tokenizer(query, return_tensors='pt')
        
    # Generate sparse embeddings for the tokenized keywords
    with torch.no_grad():
        sparse_emb = sparse_model(
            d_kwargs=tokens.to('cpu')
        )['d_rep'].squeeze()
                
    indices = sparse_emb.nonzero().squeeze().cpu().tolist()
    values = sparse_emb[indices].cpu().tolist()
                
    sparse = {'indices':indices, 'values':values}
    sparse  
    
    # Scale dense and sparse vectors
    hdense, hsparse = hybrid_scale(d_embedding, sparse, alpha=0.5)
     
    # Perform the search with scaled vectors
    results = index.query(
        vector=hdense,  # Use the scaled dense vector
        sparse_vector=hsparse,  # Use the scaled sparse vector
        top_k=top_k,
        include_metadata=True
    )
  
    # Process and return the results
    matches = []
    for match in results['matches']:
        doc_id = (match['id'])
        score = match['score']
        data = match['metadata']
        matches.append((doc_id, score, data))
        
    return matches

