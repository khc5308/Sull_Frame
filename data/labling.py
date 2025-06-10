import pygame, os, threading, json
from fn import *

pygame.init()
screen_width, screen_height = 960, 540
running = True
screen = pygame.display.set_mode((screen_width, screen_height))
file_list = os.listdir("./data/img")

# 데이터 불러오기
if os.path.exists('./data/vec2name.json'):
    with open('./data/vec2name.json', 'r', encoding='utf-8') as f: # 인코딩 추가
        raw_data = json.load(f)
        # JSON에서 불러올 때는 키가 문자열이므로 다시 튜플(int)로 변환
        # 예: "(123, 456)" -> (123, 456)
        data = {tuple(int(x) for x in k.strip('()').split(', ')): v for k, v in raw_data.items()}
else:
    data = {}

# 사용자 입력 저장용
input_data = []
last_data = []
input_ready = False

def get_inputs():
    global input_data, input_ready, last_data, i # 전역 변수 'i'는 함수 인자로 전달하는 것이 더 좋습니다.
    input_data = []
    value = input().split()
    if value == ["99"]: # 리스트 비교로 수정
        input_data = list(last_data) # last_data는 튜플이므로 리스트로 변환
    else:
        two2ten = ""
        for j in people:
            two2ten += "1" if j in value else "0"
        input_data.append(int(two2ten,2))
        two2ten = ""

        for j in album:
            two2ten += "1" if j in value else "0"
        input_data.append(int(two2ten,2))
        two2ten = ""

        for j in sort:
            two2ten += "1" if j in value else "0"
        input_data.append(int(two2ten,2))
        two2ten = ""

        for j in contents:
            two2ten += "1" if j in value else "0"
        input_data.append(int(two2ten,2))
        two2ten = ""

        for j in element:
            two2ten += "1" if j in value else "0"
        input_data.append(int(two2ten,2))

        input_data.append(date2int(file_list[i][6:14]))  # 날짜 추가

    input_ready = True
    key = tuple(input_data)  # 딕셔너리 key로 사용하기 위해 튜플로
    if key in data:
        if file_list[i] not in data[key]:
            data[key].append(file_list[i])  # 중복 방지
    else:
        data[key] = [file_list[i]]

    last_data = key # 다음 라운드를 위해 튜플 그대로 저장

processed_files = set()
for files in data.values():
    for f_name in files:
        processed_files.add(f_name)

i = 0
while i < len(file_list) and file_list[i] in processed_files:
    i += 1

while running and i < len(file_list):
    print(f"i = {i}")

    #region 이미지 배치
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
    #endregion

    #region 입력 스레드
    input_ready = False
    input_thread = threading.Thread(target=get_inputs)
    input_thread.start()
    #endregion

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

# 저장할 때 key를 문자열로 변환해서 JSON 저장
with open('./data/vec2name.json', 'w', encoding='utf-8') as f: # 인코딩 추가
    json.dump({str(k): v for k, v in data.items()}, f, indent=4, ensure_ascii=False)

pygame.quit()