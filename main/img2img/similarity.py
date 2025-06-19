# similarity.py
# vectorization.py 출력물(모델, 임베딩, 경로 파일) 로드하여 테스트 이미지와의 유사도 측정

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
from .vectorization import Embedder, load_model  # 같은 폴더에 vectorization.py가 있어야 함

# 설정
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EMBED_DIM = 512

# 쿼리 이미지와 유사도 계산 함수
def compute_similarity(query_path, model, embeddings, paths, transform, top_k=20):
    img = Image.open(query_path).convert('RGB')
    img_t = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        q_emb = model(img_t).cpu().numpy()  # [1, D]
    sims = cosine_similarity(q_emb, embeddings)[0]
    topk = np.argsort(-sims)[:top_k]

    return paths


def main(query_image):
    # 파일 경로
    model_path = './img2img/embedder.pth'
    embeddings_path = './img2img/img_embeddings.npy'
    paths_file = './img2img/img_paths.txt'

    # 1) 모델 로드
    model = load_model(path=model_path, embed_dim=EMBED_DIM)

    # 2) 임베딩 & 경로 로드
    embeddings = np.load(embeddings_path)  # [N, D]
    with open(paths_file, 'r', encoding='utf-8') as f:
        image_paths = [line.strip() for line in f]

    # 3) 전처리 정의
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    # 4) 유사도 측정
    return compute_similarity(query_image, model, embeddings, image_paths, transform)
