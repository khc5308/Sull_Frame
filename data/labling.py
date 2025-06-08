import pygame, os, threading

pygame.init()
screen_width, screen_height = 960, 540
screen = pygame.display.set_mode((screen_width, screen_height))
file_list = os.listdir("./data/img")
element = ["인원 수", "릴리", "해원", "설윤", "배이", "지우", "규진", "기타 인물", "동물 여부",
           "어린이 사진", "앞머리 유무", "장발 유무", "자막 유무", "무대 착장 여부"]

# 사용자 입력 저장용
input_data = {}
input_ready = False

def get_inputs():
    global input_data, input_ready
    input_data = {}
    for j in element:
        value = input(f"{j}: ")
        input_data[j] = value
    input_ready = True

i = 0
while i < len(file_list):
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

        screen.fill((0, 0, 0))
        screen.blit(resized_image, (x, y))
        pygame.display.update()

    # 입력 다 받으면 다음 이미지로
    i += 1
