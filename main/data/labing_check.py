import pygame
import json
import os
import sys

JSON_FILE_PATH = './main/data/people_labling.json'
IMAGE_FOLDER_PATH = './main/data/img'
# --- Pygame 화면 및 그리드 설정 ---
COLS = 5                # 한 줄에 표시할 이미지 개수 (열)
IMAGE_WIDTH = 180       # 각 이미지의 통일될 너비
IMAGE_HEIGHT = 180      # 각 이미지의 통일될 높이
PADDING = 20            # 이미지와 이미지 사이의 간격
HEADER_HEIGHT = 50      # 상단에 현재 키를 표시할 공간
TEXT_HEIGHT = 40        # 이미지 하단에 파일 이름을 표시할 공간

# 색상 정의
WHITE = (255, 255, 255)
BLACK = (30, 30, 30)
GRAY = (100, 100, 100)
BLUE = (100, 150, 255)

def run_key_based_viewer():
    """
    Pygame을 사용하여 JSON의 키(key) 단위로 이미지를 보여줍니다.
    아무 키나 누르면 다음 키의 이미지들로 전환됩니다.
    """
    # 1. JSON 데이터 로드
    try:
        with open(JSON_FILE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"❌ 오류: JSON 파일 '{JSON_FILE_PATH}'을(를) 찾을 수 없습니다.")
        return

    if not data:
        print("ℹ️ JSON 파일에 데이터가 없습니다.")
        return

    # 키 순서를 보장하기 위해 리스트로 변환
    keys = list(data.keys())
    current_key_index = 0

    # 2. Pygame 초기화
    pygame.init()
    screen_width = (IMAGE_WIDTH + PADDING) * COLS + PADDING
    screen_height = 800  # 창의 초기 높이
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("키 기반 이미지 뷰어 (아무 키나 눌러 다음으로)")

    # 폰트 설정
    header_font = pygame.font.SysFont("malgungothic", 28, bold=True)
    filename_font = pygame.font.SysFont("malgungothic", 14)

    # 3. 메인 루프
    running = True
    clock = pygame.time.Clock()

    while running:
        # 이벤트 처리 (종료, 키 입력)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 아무 키나 눌렸을 때
            if event.type == pygame.KEYDOWN:
                # 다음 키로 인덱스 변경 (마지막이면 처음으로)
                current_key_index = (current_key_index + 1) % len(keys)

        # 현재 키에 해당하는 이미지 정보 가져오기
        current_key = keys[current_key_index]
        filenames = data[current_key]
        
        # 화면 그리기
        screen.fill(BLACK)

        # 상단에 현재 키 정보 표시
        header_text = header_font.render(f"Key: {current_key}", True, BLUE)
        header_rect = header_text.get_rect(center=(screen_width / 2, HEADER_HEIGHT / 2))
        screen.blit(header_text, header_rect)

        # 현재 키의 이미지들을 격자로 표시
        for i, filename in enumerate(filenames):
            row = i // COLS
            col = i % COLS

            # 셀의 위치 계산
            cell_height = IMAGE_HEIGHT + PADDING + TEXT_HEIGHT
            x = col * (IMAGE_WIDTH + PADDING) + PADDING
            y = row * cell_height + PADDING + HEADER_HEIGHT
            
            # 이미지 불러오기 및 리사이징
            path = os.path.join(IMAGE_FOLDER_PATH, filename)
            try:
                image = pygame.image.load(path)
                image = pygame.transform.scale(image, (IMAGE_WIDTH, IMAGE_HEIGHT))
            except pygame.error:
                # 이미지 로드 실패 시 회색 상자로 대체
                image = pygame.Surface((IMAGE_WIDTH, IMAGE_HEIGHT))
                image.fill(GRAY)
            
            # 이미지 그리기
            screen.blit(image, (x, y))

            # 이미지 파일 이름 표시
            filename_text = filename_font.render(filename, True, WHITE)
            filename_rect = filename_text.get_rect(center=(x + IMAGE_WIDTH / 2, y + IMAGE_HEIGHT + TEXT_HEIGHT / 2))
            screen.blit(filename_text, filename_rect)

        pygame.display.flip() # 화면 업데이트
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    run_key_based_viewer()