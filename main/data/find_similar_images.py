# find_similar_images.py
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity

# 1. 하이퍼파라미터 설정 (모델 학습 시 사용한 것과 동일해야 함)
EMBEDDING_DIM = 256
# !!! 중요: 모든 훈련 이미지가 이 디렉토리 안에 직접 존재한다고 가정합니다.
DATA_ROOT_DIR = './img' # 훈련 이미지들이 있는 디렉토리
MODEL_PATH = 'image_embedding_model.pth' # 저장된 모델 파일 경로
BATCH_SIZE_INFERENCE = 128 # 임베딩 추출 시 사용할 배치 크기

# 2. 이미지 전처리 파이프라인 (모델 학습 시 사용한 것과 동일해야 함)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 3. 임베딩 모델 정의 (train_model.py와 동일)
class ImageEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim):
        super(ImageEmbeddingNet, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        num_ftrs = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        self.embedding_layer = nn.Linear(num_ftrs, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.embedding_layer(x)
        return x

# >>> 변경된 부분: CustomDataset for flat directory structure (train_model.py와 동일) <<<
class FlatImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []

        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                self.image_paths.append(os.path.join(root_dir, filename))
        
        if not self.image_paths:
            raise RuntimeError(f"Found 0 images in {root_dir}. Please check your DATA_ROOT_DIR and file extensions.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # 레이블은 유사성 검색에서는 필요 없지만, Dataset 요구사항 충족을 위해 반환 (더미 값)
        dummy_label = 0 
        return image, dummy_label, img_path # 이미지, 더미 레이블, 파일 경로 반환

# 4. 모델 로드 및 임베딩 추출 함수 (이전과 동일)
def load_model(model_path, embedding_dim, device):
    model = ImageEmbeddingNet(embedding_dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 평가 모드로 설정
    print(f"Model loaded from {model_path}")
    return model

def get_image_embedding(model, image_path, transform, device):
    try:
        image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model(image_tensor)
    
    return embedding.cpu().numpy()

# >>> 변경된 부분: FlatImageDataset에서 이미지 경로 추출 <<<
def extract_all_train_embeddings(model, train_dataset, device):
    """
    훈련 데이터셋의 모든 이미지에 대한 임베딩과 해당 이미지 파일 경로를 추출합니다.
    """
    print("\nExtracting embeddings for the entire training dataset...")
    all_embeddings = []
    all_image_paths = [] # 실제 이미지 파일 경로 저장

    # num_workers를 0으로 설정하면 PyTorch의 자체 오류 방지
    # for flat directory, though it might be slower
    train_loader_inference = DataLoader(train_dataset, batch_size=BATCH_SIZE_INFERENCE, shuffle=False, num_workers=4)

    with torch.no_grad():
        # FlatImageDataset은 (image_tensor, dummy_label, path) 튜플을 반환합니다.
        for i, (inputs, labels, paths) in enumerate(train_loader_inference):
            inputs = inputs.to(device)
            embeddings = model(inputs)
            all_embeddings.append(embeddings.cpu().numpy())
            
            all_image_paths.extend(paths) # 파일 경로를 리스트에 추가

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"Shape of all training embeddings: {all_embeddings.shape}")
    print(f"Number of training image paths stored: {len(all_image_paths)}")
    return all_embeddings, all_image_paths

# 5. 유사한 이미지 검색 함수 (이전과 동일)
def find_similar_images(query_image_path, model, train_embeddings, train_image_paths, transform, device, top_k=50):
    query_embedding = get_image_embedding(model, query_image_path, transform, device)
    if query_embedding is None:
        return [], []

    query_embedding = query_embedding.reshape(1, -1)

    similarities = cosine_similarity(query_embedding, train_embeddings)[0]

    most_similar_indices = np.argsort(similarities)[::-1]
    
    excluded_count = 0
    final_indices = []
    for idx in most_similar_indices:
        # 이 부분을 통해 쿼리 이미지와 동일한 경로의 이미지는 제외할 수 있습니다.
        # 실제 파일 시스템 경로를 비교합니다.
        if os.path.abspath(train_image_paths[idx]) != os.path.abspath(query_image_path):
            final_indices.append(idx)
            if len(final_indices) >= top_k:
                break
        else:
            excluded_count += 1
            if excluded_count == 1:
                print(f"Query image itself found in results (similarity 1.0) and excluded: {train_image_paths[idx]}")


    similar_image_filenames = [train_image_paths[idx] for idx in final_indices]
    similarity_scores = similarities[final_indices]
    
    return similar_image_filenames, similarity_scores

# 6. 메인 실행 블록
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 모델 로드
    model = load_model(MODEL_PATH, EMBEDDING_DIM, device)

    # 훈련 데이터셋 로드 (FlatImageDataset 사용)
    train_dataset_for_embedding_extraction = FlatImageDataset(
        root_dir=DATA_ROOT_DIR,
        transform=transform
    )
    print(f"Loaded {len(train_dataset_for_embedding_extraction)} images for embedding extraction from {DATA_ROOT_DIR}")


    # 훈련 데이터셋의 모든 임베딩 추출
    train_embeddings, train_image_paths = extract_all_train_embeddings(
        model, train_dataset_for_embedding_extraction, device
    )

    # --- 사용자 입력 이미지 경로 설정 (수정된 부분!) ---
    query_image_path = "/img/nmixx_20241010_2_2.jpg"

    # 쿼리 이미지 경로 유효성 검사
    if not os.path.exists(query_image_path):
        print(f"Error: The specified query image does not exist: {query_image_path}")
        print("Please ensure the image file is at the correct path.")
    else:
        print(f"\nFinding similar images for: {query_image_path}")
        similar_images, similarities_scores = find_similar_images(
            query_image_path, model, train_embeddings, train_image_paths, transform, device, top_k=50
        )

        print(f"\nTop 50 most similar images to {os.path.basename(query_image_path)}:")
        if similar_images:
            for i, (img_path, score) in enumerate(zip(similar_images, similarities_scores)):
                print(f"{i+1}. {os.path.basename(img_path)} (Similarity: {score:.4f})")
        else:
            print("No similar images found or query image could not be processed.")