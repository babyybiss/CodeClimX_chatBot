import os
import json
from sentence_transformers import SentenceTransformer
from splade.models.transformer_rep import Splade
from transformers import AutoTokenizer
import torch
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# sparse model
sparse_model_name = 'naver/splade-cocondenser-ensembledistil'
sparse_model = Splade(sparse_model_name, agg='max')
sparse_model.to('cpu')
sparse_model.eval()
tokenizer = AutoTokenizer.from_pretrained(sparse_model_name)

load_dotenv()

api_key = os.getenv('PINECONE_API_KEY')
if not api_key:
    raise ValueError("Please set the PINECONE_API_KEY environment variable.")

# Initialize Pinecone client
pc = Pinecone(api_key=api_key)

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


def read_and_combine_texts(directory_path):
    # 결과를 저장할 리스트 초기화
    timestamps = []
    paragraphs = []
    
    # 지정된 디렉토리 내의 모든 파일을 파일명 순으로 정렬하여 처리
    for filename in sorted(os.listdir(directory_path)):
        # 완전한 파일 경로 생성
        file_path = os.path.join(directory_path, filename)
        # 파일 확장자가 .txt인 경우만 처리
        if os.path.isfile(file_path) and file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as file:
                # 파일 내용을 JSON으로 로드
                data = json.load(file)
                # 각 항목의 timestamp와 paragraph 값을 추출하여 리스트에 추가
                for item in data:
                    timestamps.append(item['timestamp'])
                    paragraphs.append(item['paragraph'])
    return timestamps, paragraphs

def convert_times_to_seconds(time_list):
    seconds_list = []
    for time_str in time_list:
        # 시간, 분, 초를 분리
        hours, minutes, seconds = map(int, time_str.split(':'))
        # 전체 시간을 초로 환산
        total_seconds = hours * 3600 + minutes * 60 + seconds
        seconds_list.append(total_seconds)
    return seconds_list

def dense_embedding(model, texts):
    dense_embeddings = []
    for i in texts:
        embedded_text = model.encode(i)
        dense_embeddings.append(embedded_text)
    return dense_embeddings

def sparse_enbedding(tokenizer, model, texts):
    indices = []
    values = []
    for i in texts:
        tokens = tokenizer(i, return_tensors='pt')

        with torch.no_grad():
            sparse_emb = model(
                d_kwargs=tokens.to('cpu')
            )['d_rep'].squeeze()
        sparse_emb.shape
        
        indice = sparse_emb.nonzero().squeeze().cpu().tolist()
        value = sparse_emb[indice].cpu().tolist()
        indices.append(indice)
        values.append(value)
    return indices, values



# 함수 사용 예
directory_path = '/Users/babyybiss/dev/projects/codeClimX_chatbot/chatbot/data/summarization_english'
video_name = 'Machine Learning in 2024 Beginners Course'
dense_model = SentenceTransformer('distiluse-base-multilingual-cased-v1', device='cpu')
url='https://youtu.be/P0WmfImwH1Q'

timestamps, paragraphs = read_and_combine_texts(directory_path)

seconds = convert_times_to_seconds(timestamps)

dense_embeddings = dense_embedding(dense_model, paragraphs)


sparse_indices, sparse_values = sparse_enbedding(tokenizer, sparse_model, paragraphs)

# print(sparse_indices[15])

'''
Pinecone Structure : 
	index : videoid + idx
	values : distiluse-base-multilingual-cased-v1
	sparse indices
	sparse values
	text : 원문
	videoName : 저장되는 비디오명
	url : url + 재생시점
'''



vectors = []
cnt = 1
for emb, text, second, sparse_indice, sparse_value in zip(dense_embeddings, paragraphs, seconds, sparse_indices, sparse_values):
    vector = {"id": "ML2024SP" + "_" + str(cnt),
              "values" : emb,
              "sparse_values" : {'indices': sparse_indice, 'values': sparse_value},
              "metadata" : {"text": text, "videoName": video_name, "url": url, "second": second}
              }
    vectors.append(vector)
    cnt += 1
print(vector)


index.upsert(vectors=vectors)
    

