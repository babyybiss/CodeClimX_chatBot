from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
from transformers import AutoModelForMaskedLM, AutoTokenizer
from scipy import sparse

okt = Okt()

from splade.models.transformer_rep import Splade




# 문서 전처리 함수 (형태소 분석 및 토큰화)
def preprocess_text(text):
    tokens = okt.morphs(text)
    return tokens

# 코사인 유사도 계산 함수
def cos_sim(A, B):
    return np.dot(A, B.T) / (np.linalg.norm(A) * np.linalg.norm(B))

# 한글 질문과 답변
ko_question = "회귀 분석은 어떤 경우애 사용이 돼요?"
ko_answers = [
    "제 5강에서는 로지스틱 회귀(logistic regression)라는 간단하면서도 중요한 분류 모델에 대해 소개한다. 이는 출력이 확률이어야 하는 문제를 다룰 때 유용하다. 로지스틱 회귀는 사회과학, 의학, 공학 등 다양한 분야에서 널리 사용된다. 로지스틱 회귀는 주어진 변수의 데이터 세트에 대한 특정 클래스에 속하는 사건, 관측치의 조건부 확률을 모델링하는 지도 분류 기법이다. 로지스틱 회귀는 두 개의 범주를 가질 수 있지만, 이후 두 클래스를 가질 때 이상적으로 작동하는 것을 배운다. 로지스틱 회귀는 두 변수 간의 관계가 선형일 때, 종속 변수가 범주형 변수일 때, 변수를 확률의 형태로 예측하고자 할 때 유용하다. 로지스틱 회귀는 예측 과정 중에 각 관측치가 특정 클래스에 속할 확률을 예측한다. 로지스틱 회귀는 응답 변수를 예측하는 대신 특정 범주에 속하는 확률을 모델링한다. 로지스틱 회귀는 로그 확률을 예측하며, 독립 변수의 계수는 독립 변수가 한 단위 변경될 때 로그 확률의 변화를 나타낸다. 로지스틱 회귀에서 중요한 개념은 가능도(likelihood) 함수다. 가능도 함수는 관찰된 데이터 주어진 모델의 매개변수를 추정하는 데 사용된다. 가능도 함수는 관찰된 데이터의 확률을 설명하며, 버누이(PMF) 분포에 기반한 것이다. 최대 가능도 추정은 주어진 입력 데이터 및 모델이 관찰 데이터 결과의 확률을 계산하는 가능도 함수를 사용한다. 로지스틱 함수는 항상 S-형 곡선을 생성하며, 독립 변수 X의 값에 관계없이 대부분의 경우 영과 일 사이의 합리적인 추정치를 제공한다. 최대 가능도 추정의 목표는 가능도 함수를 최대화하는 일련의 추정값을 찾는 것이다. 이 과정에는 가능도 함수 정의, 로그 가능도 함수 작성, 로그 가능도 함수의 최대값 찾기 등의 단계가 포함된다. 최종적으로, 모델 적합도를 확인하고 최대 가능도 추정을 기반으로 새로운 데이터에 대한 예측을 수행하며 모델의 성능을 다양한 평가 지표로 평가한다. 로지스틱 회귀의 장점에는 모델의 단순함, 낮은 분산, 낮은 편향, 확률 제공 등이 있지만, 비선형 관계 모델링 불가, 클래스가 잘 분리된 경우 불안정, 두 개 이상의 클래스가 있는 경우 불안정 등의 단점도 있다.",
    "선형 회귀는 단순하지만 강력하며 데이터 과학 및 기계 학습 프로젝트에서 시작하기에 적합한 도구로, 데이터의 중요한 특성을 파악하고 미래를 예측하는 데 사용됩니다. 이 프로젝트는 캘리포니아 주택 가격의 동인을 찾는 것을 목표로 하며, 데이터를 정제하고 주요 경향을 시각화하는 과정을 통해 다양한 파이썬 라이브러리를 사용하여 데이터 처리 방법을 학습합니다. 선형 회귀를 파이썬에서 구현하고 데이터 과학 프로젝트를 수행하기 위해 필요한 기본 단계를 익히게 됩니다. 이 과정을 통해 pandas, scikit-learn, statsmodels, matplotlib, seaborn 등의 데이터 과학 및 기계 학습에 관련된 다양한 파이썬 라이브러리를 다루게 되며, 이 프로젝트를 개인 웹사이트와 이력서에 포함시킬 수 있게 됩니다. 예측 분석 및 인과 분석 분야의 사례 연구를 통해, 선형 회귀를 사용하여 캘리포니아 주택 가치를 정의하는 특성을 식별하는 과정을 단계별로 실습합니다. 데이터 로딩 및 처리, 누락 데이터 분석, 이상치 탐지 및 제거, 데이터 시각화, 상관 관계 분석을 포함한 체계적 접근 방식을 통해, 사례 연구의 본질을 이해하고 다중 선형 회귀를 수행하여 캘리포니아 주택 블록의 가치를 정의하는 특성을 식별합니다. 구글 클라우드와 GitHub를 활용한 코딩 과정에서 필요한 파이썬 라이브러리를 불러오고, 데이터 랭글링, 배열 처리, 통계 모델링 등의 다양한 데이터 과학 기술을 적용하며, 선형 회귀를 통한 인과 분석과 예측 분석을 위해 scikit-learn 및 statsmodels.api 라이브러리를 사용하는 방법을 학습합니다. 이 프로젝트는 데이터 과학 및 기계 학습에 대한 실습 경험을 제공하고, 학습자가 이 분야의 기본 개념과 기술을 탄탄히 다질 수 있는 기회를 제공합니다.",
    "데이터를 로드하는 과정에서 Google Cloud의 특정 폴더에 'housing.csv' 파일을 저장하였으며, 이 데이터는 지정된 페이지에서 다운로드할 수 있다. 해당 데이터(409기가바이트)를 다운로드하여 Google Cloud에 업로드하는 과정을 설명하고 있으며, 파일 경로를 복사하여 변수에 할당하는 방법을 제시한다. 이어서, pandas라이브러리의 read_csv 함수를 이용하여 데이터를 로드하는 방법을 소개한다. 데이터 탐색 단계에서는 데이터의 열(컬럼)을 확인하여 경도, 위도, 주택 중위 연령, 총 방 수, 총 침실 수, 인구수, 가구 수, 중위 소득, 주택 가격 중위값, 해안 근접성 등의 변수들이 포함되어 있음을 언급한다. 이러한 변수들의 명명방식이 파이썬에서 일반적으로 사용되는 언더스코어(_)를 포함하고 있으며, 공식 문서와는 다를 수 있지만 의미는 동일함을 설명한다. 특히, 'ocean proximity' 변수는 해안가와의 근접성을 설명하며, 이는 주택 가격에 영향을 줄 수 있는 요소임을 언급한다. 코드 실행을 통해 얻은 데이터의 상위 10행을 검토하며, 이를 통해 얻은 주요 정보를 요약한다. 이 과정은 파이썬을 이용한 선형 회귀 모델 구축에 있어 데이터 처리 및 준비의 중요한 단계임을 강조한다.",
    "특히 회귀 분석과 코사인 분석에서 변수 쌍 간의 상관 관계 점수를 계산하여 상관 관계 행렬을 얻는 과정을 말하며, 선형 회귀의 가정 중 하나는 독립 변수 간에 높은 상관 관계가 없어야 한다는 것입니다. 상관 관계 히트맵은 문제가 될 수 있는 독립 변수를 식별하는 데 유용한 방법으로, 독립 변수 간에 높은 상관 관계가 있는 경우 멀티콜리니어리티를 다루고 있을 수 있음을 의미합니다. C-burn을 사용하여 히트맵을 작성하면, 음의 상관 관계를 가리키는 밝은 색부터 긍정적인 상관 관계를 나타내는 어두운 녹색까지 다양한 색상을 볼 수 있습니다. 또한, 독립 변수는 유사한 정보를 함유하고 있기 때문에 높은 상관 관계를 가진 변수 중 하나를 제거하는 것을 고려할 수 있습니다. 총 침실 수와 세대 수는 긍정적으로 상관 관계가 높아 이 중 하나를 제거함으로써 완벽한 멀티콜리니어리티를 피할 수 있습니다. 카테고리 변수에 대해서는 원-핫 인코딩 대신 더미 변수를 생성해 선형 회귀 모델에 적용하는 것이 권장되며, 멀티콜리니어리티를 피하기 위해 하나의 카테고리를 제외시켜야 합니다. 머신 러닝 모델 훈련을 위해서는 데이터를 훈련 세트와 테스트 세트로 나누고, 모델의 훈련에는 독립 변수 세트와 종속 변수, 즉 목표 변수인 중간 주택 가격을 사용합니다. 이는 블록 내 주택의 특성이 중간 주택 가치에 미치는 영향을 식별하고자 하는 목적을 가지고 있습니다.",
    "탐색 알고리즘에는 다양한 유형이 있으며, 특히 탐욕적 최선 우선 탐색(Greedy Best First Search, GBFS)에 대해 논의한다. 해당 알고리즘은 DFS(깊이 우선 탐색)나 BFS(너비 우선 탐색)와 달리 목표에 가장 가까운 노드를 우선 확장한다. 이 알고리즘은 목표까지의 정확한 거리를 알지 못하며, 대신 목표까지의 예상 거리를 제공하는 휴리스틱 함수, 즉 h(n)을 사용하여 추정한다. 미로 해결 알고리즘에서 휴리스틱 함수는 맨해튼 거리를 사용하여 각 셀로부터 목표까지의 거리를 추정한다. GBFS는 이 휴리스틱을 기반으로 해서 더 작은 맨해튼 거리를 가진 노드를 우선적으로 탐색한다. 예를 들어, 이 알고리즘이 미로에서 특정 지점에 도달하기 위해 경로를 결정할 때, 실험적으로 목표까지의 가까운 거리를 보여주는 노드를 선택한다. 그러나 GBFS는 최적의 경로를 보장하지 않는다; 일부 경우에는 최적보다 긴 경로를 선택할 수도 있다. 이러한 문제를 해결하기 위해, 이 알고리즘의 목표는 휴리스틱을 사용해서 더 나은 결정을 내리고 상태 공간의 전체적인 탐색을 줄이는 것이지만, 동시에 알고리즘의 최적성을 달성하도록 수정하는 것이다. 알고리즘의 개선을 위해서는 경로를 따라 휴리스틱 수치가 증가하는 경우를 확인하고, 조건에 따라 더 적은 단계를 요구하는 경로를 선택하여 최적의 결과를 도출할 수 있는 방법을 고려해야 한다.",
    "A-star 탐색 알고리즘은 휴리스틱뿐만 아니라 특정 상태에 도달하기까지의 경로 길이도 고려하여 문제를 해결한다. 이 알고리즘은 노드에 도달하는데 필요한 비용(g(n))과 목표까지의 추정 거리(h(n))의 합이 최소가 되는 노드를 확장함으로써 작동한다. g(n)은 해당 노드에 도달하기까지 걸린 단계 수를, h(n)은 문제에 따라 달라지는 휴리스틱 추정치를 나타낸다. A-star 알고리즘은 두 가지 정보, 즉 현재 위치에서 목표까지 예상되는 거리뿐만 아니라 그 위치에 도달하기까지 이동한 거리도 함께 고려한다. 이 알고리즘의 핵심은 g(n)과 h(n)의 합이 최소인 경로를 선택함으로써 최적의 해결책을 찾는 것이다. 이를 위해 A-star 탐색은 휴리스틱이 접근 가능하고 일관성이 있는 경우, 즉 실제 비용을 과대 평가하지 않고 모든 단계에서 후속 노드의 휴리스틱 가치가 현재 노드의 휴리스틱 가치와 그 노드로의 단계 비용을 합한 것보다 크지 않을 때 최적의 솔루션을 찾을 수 있다. A-star 탐색은 특정 조건 하에 최적의 해를 제공하는 알고리즘이며, 사용된 휴리스틱의 품질에 따라 문제 해결의 효율성이 달라질 수 있다. 이 알고리즘은 메모리를 상당히 사용할 수 있지만, 메모리 사용을 줄이는 대안적 접근 방식과 최적화된 다른 탐색 알고리즘들도 존재한다."
]

# 영어 질문과 답변
eng_question = "What are one of the advantages of A-STAR algorithm?"
eng_doc = [
    "BFS is likely to find the shortest path to the goal, but it might require examining a large number of nodes, which can be resource-intensive.", # 낮아야 한다
    "A-STAR algorithm combines elements of Dijkstra's algorithm and greedy best-first search by using both the cost of the path from the start node and a heuristic estimate of the cost to reach the goal.", # 높아야 한다
    "Implementations can vary, including the use of object-oriented programming techniques for organizing search algorithms and their components, such as nodes, frontiers, and actions within the context of a solvable problem like a maze.", # 낮아야 한다
    "typically, A-* algorithm requires less time to find the optimal path compared to other search algorithms like Dijkstra's or breadth-first search, especially when the heuristic is well-chosen.", # 높아야 한다
    "The entire framework, by enabling the AI to anticipate several moves ahead and evaluate outcomes, forms the foundation for strategic game play in tic-tac-toe." #낮아야 한다
]

  
# 문서 전처리 및 토큰화
tokenized_corpus = [preprocess_text(doc) for doc in ko_answers]
tokenized_question = preprocess_text(ko_question)

# BM25 모델 초기화 및 쿼리에 대한 점수 계산
bm25 = BM25Okapi(tokenized_corpus)
bm25_scores = bm25.get_scores(tokenized_question)

# TF-IDF 벡터화 및 코사인 유사도 계산
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform([ko_question] + ko_answers)
question_vec = tfidf_matrix[0].toarray()
cosine_similarities = []
for i in range(1, tfidf_matrix.shape[0]):
    answer_vec = tfidf_matrix[i].toarray()
    cosine_similarities.append(cos_sim(question_vec, answer_vec)[0][0])

# SPLADE 벡터화 및 코사인 유사도 계산
sparse_model_id = 'naver/splade-cocondenser-ensembledistil'

sparse_model = Splade(sparse_model_id, agg='max')
sparse_model.to('cpu')
sparse_model.eval()

tokenizer = AutoTokenizer.from_pretrained(sparse_model_id)
tokens = tokenizer(eng_question, return_tensors='pt')
with torch.no_grad():
    sparse_emb = sparse_model(
        d_kwargs=tokens.to('cpu')
    )['d_rep'].squeeze()
sparse_emb.shape
        
indices = sparse_emb.nonzero().squeeze().cpu().tolist()
values = sparse_emb[indices].cpu().tolist()

sparse_emb = sparse_model(d_kwargs=tokens.to('cpu'))['d_rep'].squeeze()

# Convert the PyTorch tensor to a NumPy array if it's not already
if isinstance(sparse_emb, torch.Tensor):
    sparse_emb = sparse_emb.cpu().detach().numpy()

# Convert the dense NumPy array to a SciPy sparse matrix (CSR format)
sparse_matrix = sparse.csr_matrix(sparse_emb)

# Assuming you have the SPLADE model and tokenizer already set up
def generate_and_save_embeddings(documents, tokenizer, model, save_path):
    for i, doc in enumerate(documents):
        tokens = tokenizer(doc, return_tensors='pt')
        with torch.no_grad():
            embeddings = model(d_kwargs=tokens.to('cpu'))['d_rep'].squeeze()
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().detach().numpy()
            sparse_matrix = sparse.csr_matrix(embeddings)
            sparse.save_npz(f'{save_path}/doc_{i}.npz', sparse_matrix)

# Function to compute cosine similarity (as defined previously)
def compute_similarity(sparse_vec1, sparse_vec2):
    dot_product = sparse_vec1.dot(sparse_vec2.transpose()).toarray()[0, 0]
    norm1 = np.sqrt(sparse_vec1.dot(sparse_vec1.transpose()).toarray()[0, 0])
    norm2 = np.sqrt(sparse_vec2.dot(sparse_vec2.transpose()).toarray()[0, 0])
    return dot_product / (norm1 * norm2)



# 결과 출력
for i, (score, bm25_score) in enumerate(zip(cosine_similarities, bm25_scores)):
    print(f"\n\n답변 {i+1} - tf-idf 코사인 유사도: {score:.4f}, BM25 점수: {bm25_score:.4f}")

# 질문의 형태소 분석
print('-' * 100)
print(f'질문의 형태소 분석 및 토큰화 : {tokenized_question}')
print('-' * 100)
# 답변의 형태소 분분석
for i in (tokenized_corpus):
    print(f'\n\n답변의 형태소 분석 및 토큰화 {i}')


import numpy as np
from scipy import sparse

def compute_similarity(sparse_vec1, sparse_vec2):
    # Compute the dot product
    dot_product = sparse_vec1.dot(sparse_vec2.transpose()).toarray()[0, 0]
    # Compute the norms
    norm1 = np.sqrt(sparse_vec1.dot(sparse_vec1.transpose()).toarray()[0, 0])
    norm2 = np.sqrt(sparse_vec2.dot(sparse_vec2.transpose()).toarray()[0, 0])
    # Compute the cosine similarity
    similarity = dot_product / (norm1 * norm2)
    return similarity

# splade 결과
# Load the query embedding

# Generate and save embeddings for each document
generate_and_save_embeddings(eng_doc, tokenizer, sparse_model, '/Users/babyybiss/dev/projects/codeClimX_chatbot')

# Load the query embedding
query_embedding = sparse.load_npz('/Users/babyybiss/dev/projects/codeClimX_chatbot/sparse_emb.npz')

# Load and compare embeddings
similarities = []
for i in range(len(eng_doc)):
    doc_embedding = sparse.load_npz(f'/Users/babyybiss/dev/projects/codeClimX_chatbot/doc_{i}.npz')
    similarity = compute_similarity(query_embedding, doc_embedding)
    similarities.append(similarity)

# Sort and print sorted documents based on similarity
sorted_docs = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
for i in sorted_docs:
    print(f"Document {i+1} Similarity: {similarities[i]}")
