import os
from django.shortcuts import render,redirect
from django.http import JsonResponse
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .similarity import main

# Create your views here.
def img2img2_view(request):
    return render(request,"img2img.html")

def result(request):
     return render(request,"img2img_result.html")

def upload_image(request):
    if request.method != 'POST' or 'image' not in request.FILES:
        return JsonResponse({'status': 'error', 'error': '잘못된 요청'}, status=400)

    image_file = request.FILES['image']

    # 저장 경로
    upload_path = os.path.join("img2img")
    os.makedirs(upload_path, exist_ok=True)  # uploads 폴더 없으면 생성

    # 확장자 추출 → ".jpg", ".png" …
    _, ext = os.path.splitext(image_file.name)
    new_name = f"upload_img{ext.lower()}"     # 예: upload_img.jpg

    fs = FileSystemStorage(location=upload_path)
    filename = fs.save(new_name, image_file)  # 중복 시 upload_img_1.jpg처럼 자동 변경

    file_url = filename
    return JsonResponse({'status': 'success', 'url': file_url})

def result(request):
    a = main("./img2img/upload_img.jpg")
    return JsonResponse({'status': a})