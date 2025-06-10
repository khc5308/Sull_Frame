import pygame, os, threading, json
from fn import *

pygame.init()
screen_width, screen_height = 960, 540
running = True
screen = pygame.display.set_mode((screen_width, screen_height))
file_list = os.listdir("./data/img")

# 데이터 불러오기
if os.path.exists('./data/vec2name.json'):
    with open('./data/vec2name.json', 'r') as f:
        raw_data = json.load(f)
        data = {tuple(map(int, k)): v for k, v in raw_data.items()}
else:
    data = {}

# 요소 정의
element = ["릴리", "해원", "설윤", "배이", "지우", "규진", "기타 인물", "동물 여부",N
           "어린이 사진", "앞머리 유무", "장발 유무", "자막 유무", "무대 착장 여부"]

# 사용자 입력 저장용
input_data = []
last_data = []
input_ready = False

def get_inputs():
    global input_data, input_ready, last_data, i
    input_data = []
    for j in element:
        value = input(f"{j}: ")
        if value == "99":
            input_data = last_data
            break
        input_data.append(int(value))  # 숫자 입력 받기
    else:
        input_data.append(date2int(file_list[i][6:15]))  # 날짜 추가

    input_ready = True

    key = tuple(input_data)  # 딕셔너리 key로 사용하기 위해 튜플로

    if key in data:
        if file_list[i] not in data[key]:
            data[key].append(file_list[i])  # 중복 방지
    else:
        data[key] = [file_list[i]]

    last_data = input_data

i = 0
while running and i < len(file_list):
    print(f"i = {i}")
    img = pygame.image.load("./data/img/" + file_list[i]).convert_alpha()
    img_width, img_height = img.get_size()

    scale_w = screen_width / img_width
    scale_h = screen_height / img_height

    scale_factor = min(scale_w, scale_h)
    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)

    resized_image = pygame.transform.smoothscale(img, (new_width, new_height))
    x = (screen_width - new_width) // 2
    y = (screen_height - new_height) // 2

    # 입력 스레드 시작
    input_ready = False
    input_thread = threading.Thread(target=get_inputs)
    input_thread.start()

    # pygame 루프
    while not input_ready:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    running = False

        screen.fill((0, 0, 0))
        screen.blit(resized_image, (x, y))
        pygame.display.update()

    i += 1

# 저장할 때 key를 리스트로 변환해서 JSON 저장
with open('./data/vec2name.json', 'w') as f:
    json.dump({list(k): v for k, v in data.items()}, f, indent=4, ensure_ascii=False)
