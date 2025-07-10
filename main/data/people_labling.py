import os
import json
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from deepface import DeepFace
from ultralytics import YOLO 
import torch


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}")

# 폴더 경로 설정
REF_DIR = os.path.join(os.getcwd(), './main/data/train_img')
TARGET_DIR = os.path.join(os.getcwd(), './main/data/img')
OUTPUT_FILE = 'labeled_images_result.json'

KNOWN_PERSON_NAMES = ['lily', 'haewon', 'sullyoon', 'bae', 'jiwoo', 'kyujin']
DISTANCE_THRESHOLD = 0.5
MODEL_NAME = 'VGG-Face'

# YOLOv8 nano 모델 사용
try:
    face_detector = YOLO('main\data\yolov11n-face.pt') 
    face_detector.to(DEVICE)
except Exception as e:
    print(f"YOLO 모델 로딩 실패: {e}")
    print("모델 다운로드에 실패했을 수 있습니다. 인터넷 연결을 확인하세요.")
    exit()


print("\n기준 인물 데이터베이스를 생성합니다...")
reference_embeddings = defaultdict(list)

for person_name in tqdm(KNOWN_PERSON_NAMES, desc="기준 인물 처리 중"):
    person_folder = os.path.join(REF_DIR, person_name)
    if not os.path.isdir(person_folder):
        print(f"경고: '{person_name}'의 기준 폴더를 찾을 수 없습니다.")
        continue

    for filename in os.listdir(person_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filepath = os.path.join(person_folder, filename)
            try:
                img = cv2.imread(filepath)
                if img is None: continue

                detections = face_detector(img, verbose=False)

                # 탐지된 첫 번째 얼굴만 사용 (기준 이미지는 인물 1명으로 가정)
                if detections and len(detections[0].boxes) > 0:
                    box = detections[0].boxes.xyxy[0].cpu().numpy().astype(int)
                    x1, y1, x2, y2 = box
                    face_img = img[y1:y2, x1:x2]

                    # 탐지된 얼굴만 DeepFace로 임베딩 추출
                    embedding_obj = DeepFace.represent(
                        img_path=face_img,
                        model_name=MODEL_NAME,
                        enforce_detection=False # 이미 얼굴을 잘랐으므로 탐지 비활성화
                    )
                    reference_embeddings[person_name].append(embedding_obj[0]['embedding'])

            except Exception as e:
                print(f"'{filepath}' 처리 중 오류: {e}")


avg_reference_embeddings = {}
for name, embeddings in reference_embeddings.items():
    if embeddings:
        avg_reference_embeddings[name] = np.mean(embeddings, axis=0)
    else:
        print(f"경고: '{name}'에 대한 유효한 기준 데이터를 찾지 못했습니다.")

print("✅ 기준 데이터베이스 생성 완료.")



print(f"\n'{TARGET_DIR}' 폴더의 이미지를 분석합니다...")
os.makedirs(TARGET_DIR, exist_ok=True)
image_files = [f for f in os.listdir(TARGET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
results = defaultdict(list)

def find_cosine_distance(source, test):
    return 1 - (np.dot(source, test) / (np.linalg.norm(source) * np.linalg.norm(test)))

for filename in tqdm(image_files, desc="이미지 분석 중"):
    filepath = os.path.join(TARGET_DIR, filename)
    binary_flags = {name: False for name in KNOWN_PERSON_NAMES}
    binary_flags.update({'no_person': False, 'other_person': False})

    try:
        img = cv2.imread(filepath)
        if img is None:
            raise ValueError("이미지 파일을 읽을 수 없습니다.")

        # YOLO로 이미지 내 모든 얼굴 탐지
        detected_faces = face_detector(img, verbose=False)[0].boxes.xyxy.cpu().numpy().astype(int)

        if len(detected_faces) == 0:
            binary_flags['no_person'] = True
        else:
            for box in detected_faces:
                x1, y1, x2, y2 = box
                face_img = img[y1:y2, x1:x2]

                face_embedding = DeepFace.represent(
                    img_path=face_img,
                    model_name=MODEL_NAME,
                    enforce_detection=False
                )[0]['embedding']

                best_match_name, min_distance = None, float('inf')
                for name, ref_embedding in avg_reference_embeddings.items():
                    distance = find_cosine_distance(ref_embedding, face_embedding)
                    if distance < min_distance:
                        min_distance, best_match_name = distance, name

                if best_match_name and min_distance <= DISTANCE_THRESHOLD:
                    binary_flags[best_match_name] = True
                else:
                    binary_flags['other_person'] = True

    except Exception as e:
        print(f"'{filename}' 처리 중 오류: {e}")
        binary_flags['no_person'] = True

    binary_code = "".join(['1' if binary_flags[name] else '0' for name in KNOWN_PERSON_NAMES])
    binary_code += '1' if binary_flags['no_person'] else '0'
    binary_code += '1' if binary_flags['other_person'] else '0'
    results[binary_code].append(filename)


print("\n이미지 분석 완료.")
print(results)
print(f"\n총 {len(results)}개의 이진 코드가 생성되었습니다.")

with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(dict(results), f, indent=4, ensure_ascii=False)

print(f"\n결과가 '{OUTPUT_FILE}' 파일에 저장되었습니다.")