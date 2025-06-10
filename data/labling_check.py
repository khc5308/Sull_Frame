import pygame, os, threading, json
from fn import * # 필요한 함수가 'fn'에 있다고 가정합니다.

pygame.init()
screen_width, screen_height = 960, 540
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Pygame 이미지 뷰어") # 윈도우 제목 추가

with open('./data/vec2name.json', 'r', encoding='utf-8') as f:
    raw_data = json.load(f)
    data = {tuple(int(x) for x in k.strip('()').split(', ')): v for k, v in raw_data.items()}

# 표시할 모든 이미지를 리스트로 준비합니다.
all_images = []
for vec_key, image_names in data.items():
    for image_name in image_names:
        all_images.append((vec_key, image_name))

current_image_index = 0
total_images = len(all_images)

# 이미지를 로드하고 크기를 조정하는 함수
def load_and_scale_image(image_path):
    img = pygame.image.load(image_path).convert_alpha()
    img_width, img_height = img.get_size()

    scale_w = screen_width / img_width
    scale_h = screen_height / img_height

    scale_factor = min(scale_w, scale_h)
    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)

    resized_image = pygame.transform.smoothscale(img, (new_width, new_height))
    x = (screen_width - new_width) // 2
    y = (screen_height - new_height) // 2
    return resized_image, (x, y)

# 초기 이미지 로드
if total_images > 0:
    current_vec_key, current_image_name = all_images[current_image_index]
    image_path = os.path.join("./data/img", current_image_name)
    resized_image, image_pos = load_and_scale_image(image_path)
else:
    resized_image = None
    image_pos = (0, 0) # 이미지가 없는 경우 기본 위치

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # 다음 이미지로 이동
                current_image_index += 1
                if current_image_index >= total_images:
                    running = False # 모든 이미지를 표시했으면 종료합니다.
                    print("모든 이미지를 표시했습니다!")
                else:
                    current_vec_key, current_image_name = all_images[current_image_index]
                    image_path = os.path.join("./data/img", current_image_name)
                    resized_image, image_pos = load_and_scale_image(image_path)
                    print(f"표시 중: {current_vec_key} - {current_image_name}")

            if event.key == pygame.K_1:
                if current_image_index < total_images:
                    print(current_image_name)

    # 그리기
    screen.fill((0, 0, 0)) # 배경을 검은색으로 채웁니다.
    if resized_image: # 이미지가 로드되었을 경우에만 그립니다.
        screen.blit(resized_image, image_pos)
    pygame.display.flip() # 전체 화면을 업데이트하려면 flip()을 사용합니다.

pygame.quit()
exit()