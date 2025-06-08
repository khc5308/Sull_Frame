import pygame,os,pickle,numpy,time

pygame.init()
screen_width, screen_height = 960,540
screen = pygame.display.set_mode((screen_width,screen_height))
file_list = os.listdir("./data/img")
element = ["인원 수","릴리","해원","설윤","배이","지우","규진","기타 인물","동물 여부"
            ,"어린이 사진","앞머리 유무","장발 유뮤","자막 유무","무대 착장 여부"]
# 게시일 ( 2022.1.1일 = 0 )

i = 0
while 1:

    img = pygame.image.load("./data/img/"+file_list[i]).convert_alpha()
    img_width, img_height = img.get_size()

    scale_w = screen_width / img_width
    scale_h = screen_height / img_height

    scale_factor = min(scale_w, scale_h)

    new_width = int(img_width * scale_factor)
    new_height = int(img_height * scale_factor)

    # 이미지 리사이징
    resized_image = pygame.transform.smoothscale(img, (new_width, new_height))

    # 중앙에 위치시키기
    x = (screen_width - new_width) // 2
    y = (screen_height - new_height) // 2

    screen.fill((0, 0, 0))
    screen.blit(resized_image, (x, y)) 
    pygame.display.update()
    

    for i in 

    # user_input = input()
    # if user_input == "stop":
    #     break




    i+=1
    
