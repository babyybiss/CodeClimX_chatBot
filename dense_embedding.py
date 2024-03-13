from sentence_transformers import SentenceTransformer
import numpy as np

# 코사인 유사도 계산 함수
def cos_sim(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# 모델 리스트 초기화
ko_models = {
    "Model1": SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'),
    "Model2": SentenceTransformer('intfloat/e5-large-v2'),
    "Model3": SentenceTransformer('jhgan/ko-sbert-nli'),
    "Model4": SentenceTransformer('kakaobank/kf-deberta-base')
}

eng_models = {
    "Model1": SentenceTransformer('msmarco-bert-base-dot-v5')
}

# 한국어 질문과 답변
ko_question = "인공지능의 미래에 대한 긍정적인 측면에 대해 어떻게 생각하나요?"
ko_doc = [
    "인공지능의 미래는 매우 밝다고 생각합니다. 기술이 발전함에 따라 우리의 일상생활과 산업 전반에 걸쳐 혁신을 가져올 것입니다.", # 높아야 한다.
    "인공지능 기술은 지속적으로 발전하고 있으며, 이는 곧 우리 사회와 경제에 긍정적인 변화를 이끌 것입니다.", # 높아야 한다.
    "미래의 인공지능은 의료, 교육, 제조업 등 다양한 분야에서 인간의 능력을 확장시키는 역할을 할 것입니다.", # 높아야 한다.
    "인공지능이 발전함에 따라 실업률이 증가할 것이라는 우려가 있습니다. 이에 대한 대비책이 필요합니다.", # 낮아야 한다.
    "인공지능 기술의 윤리적인 사용에 대한 논의가 필요합니다. 기술의 발전이 인간의 가치와 충돌하지 않도록 해야 합니다.", # 낮아야 한다.
    "인공지능 기술의 발전은 사생활 침해와 같은 부정적인 측면도 가지고 있습니다. 이에 대한 사회적 합의가 필요합니다." # 낮아야 한다.
]

# 영어 질문과 답변
eng_question = "What are one of the advantages of A-STAR algorithm?"
eng_doc = [
    "A-STAR is a popular pathfinding algorithm commonly used in robotics, video games, and map routing applications due to its efficiency and accuracy in finding the shortest path.", # 낮아야 한다
    "A-STAR algorithm combines elements of Dijkstra's algorithm and greedy best-first search by using both the cost of the path from the start node and a heuristic estimate of the cost to reach the goal.", # 낮아야 한다
    "A-STAR guarantees the shortest path under certain conditions, specifically when the heuristic function used is admissible", # 낮아야 한다
    "The efficiency of A-STAR depends heavily on the quality of the heuristic function. A good heuristic can significantly reduce the number of nodes expanded during the search process, leading to faster computation.", # 높아야 한다
    "One of the key advantages of A-STAR over other search algorithms is its ability to incorporate domain-specific knowledge through the heuristic function, allowing it to find optimal solutions more efficiently in many cases." #높아야 한다
]

# 각 모델에 대해 질문과 답변들의 코사인 유사도 계산
for model_name, model in ko_models.items():
    question_embedding = model.encode(ko_question)
    answer_embeddings = model.encode(ko_doc)
    cosine_similarities = [cos_sim(question_embedding, answer_embedding) for answer_embedding in answer_embeddings]
    
    print(f"{model_name} 한글 코사인 유사도:")
    for i, score in enumerate(cosine_similarities):
        print(f"  답변 {i+1}: {score:.4f}")
    print()


# 각 모델에 대해 질문과 답변들의 코사인 유사도 계산
for model_name, model in eng_models.items():
    question_embedding = model.encode(eng_question)
    answer_embeddings = model.encode(eng_doc)
    cosine_similarities = [cos_sim(question_embedding, answer_embedding) for answer_embedding in answer_embeddings]
    
    print(f"{model_name} 영어 코사인 유사도:")
    for i, score in enumerate(cosine_similarities):
        print(f"  답변 {i+1}: {score:.4f}")
    print()