from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from konlpy.tag import Okt
from transformers import AutoTokenizer
from splade.models.transformer_rep import Splade
from scipy import sparse
import torch

# Initialize Sentence Transformers
ko_models = {
    "Model1": SentenceTransformer('sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens'),
    "Model2": SentenceTransformer('intfloat/e5-large-v2'),
    "Model3": SentenceTransformer('jhgan/ko-sbert-nli'),
    "Model4": SentenceTransformer('kakaobank/kf-deberta-base'),
    "Model5": SentenceTransformer('distiluse-base-multilingual-cased-v1'),
    "Model6": SentenceTransformer('jhgan/ko-sroberta-multitask'),
    "Model7": SentenceTransformer('upskyy/kf-deberta-multitask')
}

eng_models = {
    "Model1": SentenceTransformer('msmarco-bert-base-dot-v5'),
    "Model2": SentenceTransformer('distiluse-base-multilingual-cased-v1')
}


# Initialize SPLADE
sparse_model_id = 'naver/splade-cocondenser-ensembledistil'
sparse_model = Splade(sparse_model_id, agg='max')
sparse_model.to('cpu')
sparse_model.eval()
tokenizer = AutoTokenizer.from_pretrained(sparse_model_id)

# 코사인 유사도 계산 함수
def cos_sim(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

# Function to generate and save embeddings for documents
def generate_and_save_embeddings(documents, tokenizer, model, save_path, eng_question):
    for i, doc in enumerate(documents):
        tokens = tokenizer(doc, return_tensors='pt')
        with torch.no_grad():
            embeddings = model(d_kwargs=tokens.to('cpu'))['d_rep'].squeeze()
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().detach().numpy()
            sparse_matrix = sparse.csr_matrix(embeddings)
            sparse.save_npz(f'{save_path}/doc_{i}.npz', sparse_matrix)
    tokens = tokenizer(eng_question, return_tensors='pt')
    with torch.no_grad():
        embeddings = model(d_kwargs=tokens.to('cpu'))['d_rep'].squeeze()
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().detach().numpy()
        sparse_matrix = sparse.csr_matrix(embeddings)
        sparse.save_npz(f'{save_path}/query.npz', sparse_matrix)

# Cosine similarity function for sparse vectors
def compute_similarity(sparse_vec1, sparse_vec2):
    dot_product = sparse_vec1.dot(sparse_vec2.transpose()).toarray()[0, 0]
    norm1 = np.sqrt(sparse_vec1.dot(sparse_vec1.transpose()).toarray()[0, 0])
    norm2 = np.sqrt(sparse_vec2.dot(sparse_vec2.transpose()).toarray()[0, 0])
    similarity = dot_product / (norm1 * norm2)
    return similarity

# Function to calculate dense cosine similarities
def calculate_dense_similarities(question, documents, model):
    question_embedding = model.encode(question)
    document_embeddings = model.encode(documents)
    similarities = [cos_sim(question_embedding, doc_emb) for doc_emb in document_embeddings]
    return similarities

# Function to combine dense and sparse similarities
def combine_similarities(dense_similarities, sparse_similarities, alpha=0.5):
    # Assuming dense and sparse similarities are lists of the same length
    combined_similarities = [alpha * dense + (1 - alpha) * sparse for dense, sparse in zip(dense_similarities, sparse_similarities)]
    return combined_similarities


# Function to evaluate models
def evaluate_models(ko_question, ko_doc, eng_question, eng_doc, save_path):
        
    # Dense Embedding Similarities for Korean
    ko_dense_similarities = {}
    for model_name, model in ko_models.items():
        question_embedding = model.encode(ko_question)
        answer_embeddings = model.encode(ko_doc)
        cosine_similarities = [cos_sim(question_embedding, answer_embedding) for answer_embedding in answer_embeddings]
        
        ko_dense_similarities[model_name] = cosine_similarities
        
        print(f"{model_name} 한글 코사인 유사도:")
        for i, score in enumerate(cosine_similarities):
            print(f"  답변 {i+1}: {score:.4f}")

    # Sparse Embedding Similarities for Korean (e.g., using TF-IDF or BM25)
    okt = Okt()
    tokenized_corpus = [okt.morphs(doc) for doc in ko_doc]
    tokenized_question = okt.morphs(ko_question)

    bm25 = BM25Okapi(tokenized_corpus)
    ko_bm25_scores = bm25.get_scores(tokenized_question)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([ko_question] + ko_doc)
    question_vec = tfidf_matrix[0].toarray()
    ko_tfidf_similarities = []
    for i in range(1, tfidf_matrix.shape[0]):
        answer_vec = tfidf_matrix[i].toarray()
        ko_tfidf_similarities.append(np.dot(question_vec, answer_vec.T) / (np.linalg.norm(question_vec) * np.linalg.norm(answer_vec)))

    # Combine Dense and Sparse Similarities for Korean
    # For simplicity, we average the dense model similarities and combine with BM25
    # Feel free to use more sophisticated combination techniques
    ko_combined_similarities = []
    for i in range(len(ko_doc)):
        dense_avg = np.mean([ko_dense_similarities[model_name][i] for model_name in ko_models])
        combined_score = (dense_avg + ko_bm25_scores[i]) / 2
        ko_combined_similarities.append(combined_score)

    # Sort and print sorted Korean documents based on combined similarity
    ko_sorted_docs = sorted(range(len(ko_combined_similarities)), key=lambda i: ko_combined_similarities[i], reverse=True)
    for i in ko_sorted_docs:
        print(f"\n한글 문서 {i+1} Hybrid Similarity: {ko_combined_similarities[i]}")

    # Evaluate dense models for English
    for model_name, model in eng_models.items():
        question_embedding = model.encode(eng_question)
        answer_embeddings = model.encode(eng_doc)
        cosine_similarities = [cos_sim(question_embedding, answer_embedding) for answer_embedding in answer_embeddings]
        
        print(f"\n{model_name} 영어 코사인 유사도:")
        for i, score in enumerate(cosine_similarities):
            print(f"  답변 {i+1}: {score:.4f}")
    

    # TF-IDF and BM25 for Korean
    okt = Okt()
    tokenized_corpus = [okt.morphs(doc) for doc in ko_doc]
    tokenized_question = okt.morphs(ko_question)

    bm25 = BM25Okapi(tokenized_corpus)
    bm25_scores = bm25.get_scores(tokenized_question)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([ko_question] + ko_doc)
    question_vec = tfidf_matrix[0].toarray()
    cosine_similarities = []
    for i in range(1, tfidf_matrix.shape[0]):
        answer_vec = tfidf_matrix[i].toarray()
        cosine_similarities.append(np.dot(question_vec, answer_vec.T) / (np.linalg.norm(question_vec) * np.linalg.norm(answer_vec)))

    # SPLADE for English
    generate_and_save_embeddings(eng_doc, tokenizer, sparse_model, '/Users/babyybiss/dev/projects/codeClimX_chatbot/hybrid_test', eng_question)
    
    
    # Load the query embedding
    query_embedding = sparse.load_npz('/Users/babyybiss/dev/projects/codeClimX_chatbot/hybrid_test/query.npz')

    similarities = []
    for i in range(len(eng_doc)):
        doc_embedding = sparse.load_npz(f'/Users/babyybiss/dev/projects/codeClimX_chatbot/hybrid_test/doc_{i}.npz')
        similarity = compute_similarity(query_embedding, doc_embedding)
        similarities.append(similarity)

    sorted_docs = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)
    print()
    for i in sorted_docs:
        print(f"Document {i+1} Sparse Similarity: {similarities[i]}")
        
        # Calculate dense similarities for an English model
    dense_similarities = calculate_dense_similarities(eng_question, eng_doc, eng_models["Model1"])

    # Load the query embedding for SPLADE and calculate sparse similarities
    query_embedding = sparse.load_npz(f'{save_path}/query.npz')
    sparse_similarities = []
    for i in range(len(eng_doc)):
        doc_embedding = sparse.load_npz(f'{save_path}/doc_{i}.npz')
        similarity = compute_similarity(query_embedding, doc_embedding)
        sparse_similarities.append(similarity)
        
    # Combine dense and sparse similarities
    combined_similarities = combine_similarities(dense_similarities, sparse_similarities)    
    
    # Sort and print sorted documents based on combined similarity
    sorted_docs = sorted(range(len(combined_similarities)), key=lambda i: combined_similarities[i], reverse=True)
    print()
    for i in sorted_docs:
        print(f"Document {i+1} hybrid Similarity: {combined_similarities[i]}")


# Call the evaluation function with your questions and documents
ko_question = "태양 에너지를 사용하는 것의 이점은 무엇인가요?"
ko_doc = [
    "태양 에너지는 깨끗하고 재생 가능한 에너지원을 제공합니다.", # 높아야 한다.
    "태양광 패널 설치에는 넓은 면적이 필요한데, 이는 밀집된 도시에서는 도전이 될 수 있습니다.", # 낮아야 한다.
    "태양광 에너지 사용은 화석 연료에 대한 의존도를 줄이고 온실가스 배출을 낮춥니다.", # 높아야 한다.
    "태양 에너지 시스템의 초기 비용이 상당히 높을 수 있어 일부 가정에서는 접근하기 어려울 수 있습니다.", # 낮아야 한다.
    "인공지능 기술의 윤리적 사용에 대한 논의가 필요합니다. 기술의 발전이 인간의 가치와 충돌해서는 안 됩니다.", # 낮아야 한다.
] # Your Korean documents here
eng_question = "What are the benefits of using solar energy?"
eng_doc = [
    "Solar energy provides a clean and renewable source of power.", # Should be high.
    "Solar panels require a large area for installation, which can be a challenge in densely populated cities.", # Should be low.
    "Using solar power reduces dependence on fossil fuels and lowers greenhouse gas emissions.", # Should be high.
    "The initial cost of solar energy systems can be quite high, making it less accessible for some households.", # Should be low.
    "There is a need for discussion on the ethical use of artificial intelligence technology. The advancement of technology must not conflict with human values.", # Should be low.
    ] # Your English documents here

evaluate_models(ko_question, ko_doc, eng_question, eng_doc, '/Users/babyybiss/dev/projects/codeClimX_chatbot/hybrid_test')
