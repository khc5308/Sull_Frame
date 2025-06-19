# vectorization.py
# main/data/img 폴더 내 모든 이미지 파일을 임베딩 추출 → 저장 스크립트

import os
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms, models

# 설정
BATCH_SIZE = 64
EMBED_DIM = 512
NUM_WORKERS = 4
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 커스텀 데이터셋
class ImageDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, path

# 임베딩 모델 정의
class Embedder(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        backbone = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(2048, embed_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)

# 모델 저장
def save_model(model, path='embedder.pth'):
    torch.save(model.state_dict(), path)

# 모델 로드
def load_model(path='embedder.pth', embed_dim=EMBED_DIM):
    model = Embedder(embed_dim).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

# 메인 실행부
if __name__ == '__main__':
    base = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base,'img')

    # 이미지 수집
    exts = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
    image_paths = []
    for ext in exts:
        image_paths.extend(glob.glob(os.path.join(data_dir, '**', ext), recursive=True))
    total = len(image_paths)
    if total == 0:
        raise FileNotFoundError(f"No images found in {data_dir}")
    print(f"Found {total} images")

    # 전처리 정의
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(image_paths, transform)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = Embedder(EMBED_DIM).to(DEVICE)
    model.eval()

    embeddings = []
    saved_paths = []
    with torch.no_grad():
        for imgs, paths in loader:
            imgs = imgs.to(DEVICE)
            embs = model(imgs).cpu().numpy()
            embeddings.append(embs)
            saved_paths.extend(paths)
            print(f"Progress: {len(saved_paths)} / {total}")

    # 저장
    np.save('img_embeddings.npy', np.concatenate(embeddings, axis=0))
    with open('img_paths.txt', 'w', encoding='utf-8') as f:
        for p in saved_paths:
            f.write(p + '\n')
    save_model(model)
    print("Embedding, paths, and model saved.")
