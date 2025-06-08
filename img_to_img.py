import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess_image(img_path, target_size=(224, 224)):
    """Loads and preprocesses an image."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return preprocess_input(img_array)

def compute_embedding(model, img_path):
    """Computes the embedding of an image using the given model."""
    img = load_and_preprocess_image(img_path)
    return model.predict(img).flatten()

def find_most_similar_image(input_image_path, embeddings_path, model_path):
    """
    Finds the most similar image to the input image using precomputed embeddings.
    
    Args:
        input_image_path (str): Path to the input image.
        embeddings_path (str): Path to the precomputed embeddings.
        model_path (str): Path to the pre-trained embedding model.
    
    Returns:
        str: Path to the most similar image.
        float: Similarity score.
    """
    # Load the saved embedding model and precomputed embeddings
    model = load_model(model_path)
    embeddings = np.load(embeddings_path, allow_pickle=True).item()

    # Compute embedding for the input image
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
    # Paths
    model_path = "./model/embedding_model.h5"  # Path to the saved model
    input_image_path = path
    embeddings_path = "./model/image_embeddings.npy"  # Path to the saved embeddings

    # Find the most similar image
    similar_image, similarity_score = find_most_similar_image(
        input_image_path, embeddings_path, model_path
    )
    with open("./data/instagram_posts.json", "r", encoding="utf-8") as file:
        data = json.load(file)

    print(f"Similarity score: {similarity_score}")
    print("")
    #print(f"가장 비슷한 이미지: {similar_image}")
    print("인스타그램 링크 : ",data[similar_image[16:]])

    if similarity_score >= 0.9:
        print("인스타 사진입니다.")
    else:
        print("버블 사진이 의심됩니다.")

print("\n\n")
print("인스타 사진 검색 시스템입니다")
print("파일을 업로드하고, 이미지 주소를 입력해 검색하세요")
print("1 을 입력하면 종료됩니다")
print("\n\n")

while 1:
    a = input("검색하고 싶은 이미지의 링크를 입력하세요 : ")
    if a == "1":
        break
    main(a)
    # try:
    # except:
    #     print("올바르지 않은 경로입니다.")