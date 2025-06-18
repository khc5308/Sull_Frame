import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# 1. 하이퍼파라미터 설정
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 10
EMBEDDING_DIM = 256  # 임베딩 벡터의 차원
DATA_DIR = './data'

# 2. 데이터셋 준비 및 전처리
# 이미지 전처리 파이프라인 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ResNet 입력 크기에 맞게 리사이즈
    transforms.ToTensor(),          # 이미지를 Tensor로 변환
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # ImageNet 통계로 정규화
])

# CIFAR-10 데이터셋 로드 (예시)
# 실제 학습에서는 본인의 데이터셋으로 교체해야 합니다.
# 예: custom_dataset = MyCustomDataset(root_dir='your_image_folder', transform=transform)
train_dataset = datasets.CIFAR10(root=DATA_DIR, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root=DATA_DIR, train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 3. 임베딩 모델 정의
class ImageEmbeddingNet(nn.Module):
    def __init__(self, embedding_dim):
        super(ImageEmbeddingNet, self).__init__()
        # Pre-trained ResNet-50 로드
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 마지막 Fully Connected (FC) 레이어 제거
        # ResNet-50의 마지막 분류 레이어는 'fc' 입니다.
        num_ftrs = self.backbone.fc.in_features
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1]) # 마지막 FC 레이어 제거

        # 임베딩을 위한 새로운 FC 레이어 추가
        self.embedding_layer = nn.Linear(num_ftrs, embedding_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1) # (Batch_size, num_ftrs, 1, 1) -> (Batch_size, num_ftrs)
        x = self.embedding_layer(x)
        return x

# 4. 모델, 손실 함수, 옵티마이저 초기화
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = ImageEmbeddingNet(EMBEDDING_DIM).to(device)

# 임베딩 학습에 적합한 손실 함수 선택
# 여기서는 분류 예시를 위해 CrossEntropyLoss를 사용합니다.
# 실제 임베딩 학습에서는 ContrastiveLoss, TripletLoss, ArcFaceLoss 등이 주로 사용됩니다.
# 예시로, 임베딩을 통해 분류를 수행한다고 가정하고 CrossEntropyLoss를 사용합니다.
# 만약 동일성/유사성 임베딩을 목표로 한다면 아래 loss를 고려해야 합니다.
# criterion = nn.TripletMarginLoss()
# criterion = nn.MSELoss() # Autoencoder 기반의 임베딩 학습 시
criterion = nn.CrossEntropyLoss() # 분류를 통한 임베딩 학습 시 (Softmax with Triplet loss와 유사)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 5. 학습 루프
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    model.train() # 모델을 학습 모드로 설정
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad() # 그라디언트 초기화

            # 순전파
            # CrossEntropyLoss를 사용하기 위해 임베딩 후 선형 분류 레이어를 추가합니다.
            # 실제 임베딩 추출 시에는 이 분류 레이어를 사용하지 않습니다.
            # 여기서는 임베딩의 '유용성'을 분류 성능으로 간접적으로 측정하는 방식입니다.
            outputs = model(inputs)
            
            # CIFAR-10 클래스 개수에 맞게 final FC layer 추가
            # 모델 정의 시 embedding_layer 뒤에 최종 분류 레이어를 추가해야 합니다.
            # 아니면, Softmax + CrossEntropyLoss 대신 다른 임베딩 손실 함수를 사용해야 합니다.
            # 이 예시에서는 단순화를 위해, 임베딩 자체를 분류 입력으로 사용한다고 가정합니다.
            # 실제로는 임베딩 후 classifier = nn.Linear(EMBEDDING_DIM, num_classes)를 추가하고,
            # outputs = classifier(outputs)를 수행해야 합니다.
            
            # 이 예시 코드를 Contrastive Loss 또는 Triplet Loss와 함께 사용하려면,
            # 데이터셋에서 페어나 트리플렛을 생성하는 방식으로 변경해야 합니다.
            
            # 임시 방편으로, 현재 모델이 임베딩을 출력하고, 이 임베딩을 사용하여 분류를 수행하는 형태를 가정합니다.
            # 이를 위해 임베딩 차원(EMBEDDING_DIM)이 클래스 수와 같거나,
            # 임베딩 뒤에 분류 레이어가 붙어있다고 가정해야 합니다.
            # 현재 ImageEmbeddingNet은 embedding_dim 차원의 벡터를 출력합니다.
            # 만약 CrossEntropyLoss를 사용하려면, 이 embedding_dim이 num_classes여야 합니다.
            # 또는 아래와 같이 임시 분류 레이어를 추가해야 합니다.

            # === 수정: CrossEntropyLoss를 사용하기 위한 임시 분류 레이어 추가 ===
            num_classes = len(train_dataset.classes)
            temp_classifier = nn.Linear(EMBEDDING_DIM, num_classes).to(device)
            logits = temp_classifier(outputs)
            loss = criterion(logits, labels)
            # =================================================================

            # 역전파 및 최적화
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# 모델 학습 실행
print("\nStarting model training...")
train_model(model, train_loader, criterion, optimizer, NUM_EPOCHS)
print("Training finished.")

# 6. 학습된 모델로부터 이미지 임베딩 추출 함수
def get_image_embedding(model, image_path, transform, device):
    model.eval() # 모델을 평가 모드로 설정 (dropout, batchnorm 등 비활성화)
    
    # 이미지 로드 및 전처리
    from PIL import Image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device) # 배치 차원 추가

    with torch.no_grad(): # 그라디언트 계산 비활성화
        embedding = model(image_tensor)
    
    return embedding.cpu().numpy() # CPU로 이동 후 NumPy 배열로 반환

# 예시: 임의의 이미지에 대한 임베딩 추출 (실행하려면 이미지 파일 경로를 제공해야 함)
# try:
#     # 실제 이미지 파일 경로로 변경하세요.
#     sample_image_path = "path/to/your/image.jpg" 
#     embedding_vector = get_image_embedding(model, sample_image_path, transform, device)
#     print(f"\nEmbedding for {sample_image_path}:")
#     print(embedding_vector)
#     print(f"Embedding dimension: {embedding_vector.shape}")
# except FileNotFoundError:
#     print(f"\nError: Image file not found at {sample_image_path}. Please provide a valid path to test embedding extraction.")
# except Exception as e:
#     print(f"\nAn error occurred during embedding extraction: {e}")


# 7. (선택적) 테스트 데이터셋으로 임베딩 테스트 (유사성 측정 등)
# 실제 임베딩의 성능은 유사한 이미지 간의 거리, 클러스터링 등으로 측정됩니다.
# 여기서는 간단히 테스트 데이터셋의 첫 몇 개 이미지에 대한 임베딩을 추출하는 예시입니다.
print("\nExtracting embeddings for test dataset samples...")
model.eval()
test_embeddings = []
test_labels = []

with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(device)
        embeddings = model(inputs)
        test_embeddings.append(embeddings.cpu().numpy())
        test_labels.append(labels.cpu().numpy())
        if i * BATCH_SIZE >= 100: # 첫 100개 이미지까지만 추출
            break

test_embeddings = np.concatenate(test_embeddings, axis=0)
test_labels = np.concatenate(test_labels, axis=0)

print(f"Shape of extracted test embeddings: {test_embeddings.shape}")
print(f"Shape of extracted test labels: {test_labels.shape}")

# 임베딩 시각화 (PCA 또는 t-SNE 등을 사용할 수 있지만 여기서는 간단히 스킵)
# plt.figure(figsize=(10, 8))
# # 예를 들어, 첫 두 주성분으로 차원 축소 후 시각화
# from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# reduced_embeddings = pca.fit_transform(test_embeddings)
# plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], c=test_labels, cmap='viridis', alpha=0.6)
# plt.colorbar(label='Class Label')
# plt.title('2D PCA of Image Embeddings')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.show()