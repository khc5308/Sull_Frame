import json
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """이미지를 불러오고 전처리합니다."""
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet 기준
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert('RGB')
    img_tensor = preprocess(img).unsqueeze(0)  # 배치 차원 추가
    return img_tensor.to(device)

def compute_embedding(model, img_path):
    """이미지의 임베딩을 계산합니다."""
    model.eval()
    with torch.no_grad():
        img_tensor = load_and_preprocess_image(img_path)
        embedding = model(img_tensor).cpu().numpy().flatten()
    return embedding

def find_most_similar_image(input_image_path, embeddings_path, model_path=None):
    """
    입력 이미지와 가장 유사한 이미지를 찾습니다.
    """
    # ResNet50을 임베딩 모델로 사용하되, 마지막 fc 레이어 제거
    resnet = models.resnet50(pretrained=True)
    model = nn.Sequential(*list(resnet.children())[:-1])  # Global AvgPool 출력
    model.to(device)

    # 사전 계산된 임베딩 불러오기
    embeddings = np.load(embeddings_path, allow_pickle=True).item()

    # 입력 이미지 임베딩 계산
    input_embedding = compute_embedding(model, input_image_path)

    max_similarity = -1
    most_similar_image_path = None

    for img_path, img_embedding in embeddings.items():
        similarity = cosine_similarity(
            [input_embedding], [img_embedding]
        )[0][0]
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_image_path = img_path

    return most_similar_image_path, max_similarity

def main(path):
    input_image_path = path
    embeddings_path = "./model/image_embeddings.npy"

    similar_image, similarity_score = find_most_similar_image(
        input_image_path, embeddings_path
    )

    print(similar_image)
    print(f"Similarity score: {similarity_score}")

    if similarity_score >= 0.9:
        print("인스타 사진입니다.")
    else:
        print("버블 사진이 의심됩니다.")

print("\n\n")
print("인스타 사진 검색 시스템입니다")
print("파일을 업로드하고, 이미지 주소를 입력해 검색하세요")
print("1 을 입력하면 종료됩니다")
print("\n\n")

while True:
    a = input("검색하고 싶은 이미지의 링크를 입력하세요 : ")
    if a == "1":
        break
    main(a)
