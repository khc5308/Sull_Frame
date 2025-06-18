import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm  # ✅ 진행 바 라이브러리

def load_and_preprocess_image(img_path, device, target_size=(224, 224)):
    preprocess = transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet 평균
            std=[0.229, 0.224, 0.225]    # ImageNet 표준편차
        )
    ])
    img = Image.open(img_path).convert("RGB")
    img_tensor = preprocess(img).unsqueeze(0)
    return img_tensor.to(device)

def compute_embedding(model, img_path, device):
    img_tensor = load_and_preprocess_image(img_path, device)
    with torch.no_grad():
        embedding = model(img_tensor).squeeze().cpu().numpy()
    return embedding

def save_embeddings(model, images_dir, output_path, device):
    model.eval()
    embeddings = {}

    image_files = [
        file_name for file_name in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, file_name))
    ]

    for file_name in tqdm(image_files, desc="Processing images"):
        file_path = os.path.join(images_dir, file_name)
        try:
            embedding = compute_embedding(model, file_path, device)
            embeddings[file_path] = embedding
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    np.save(output_path, embeddings)
    print(f"✅ Embeddings saved to {output_path}")

if __name__ == "__main__":
    model_save_path = "embedding_model.pth"
    images_dir = "./data/img/"
    embeddings_output_path = "image_embeddings.npy"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model = torch.nn.Sequential(*list(model.children())[:-1])  # FC 제거
    model.to(device)

    torch.save(model.state_dict(), model_save_path)
    print(f"✅ Model saved to {model_save_path}")

    save_embeddings(model, images_dir, embeddings_output_path, device)
