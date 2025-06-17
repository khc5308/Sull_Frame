# tag2tag/views.py

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt # 개발용: CSRF 토큰 없이 POST 허용
from django.conf import settings
import os
import uuid # 고유한 파일 이름을 위해 사용
from PIL import Image # 이미지 처리를 위해 Pillow 라이브러리 필요 (pip install Pillow)
import logging

logger = logging.getLogger(__name__)

# ⭐ 파일 업로드 처리 뷰 (POST 요청) ⭐
# @csrf_exempt는 개발 편의를 위한 것으로, 실제 프로덕션 환경에서는 CSRF 보호를 활성화하고
# 프론트엔드에서 CSRF 토큰을 전송해야 합니다.
@csrf_exempt
def upload_image(request):
    if request.method == 'POST' and request.FILES.get('image'):
        uploaded_file = request.FILES['image']

        # 파일 확장자 검증 (선택 사항이지만 권장)
        allowed_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        if file_extension not in allowed_extensions:
            logger.warning(f"Attempted to upload disallowed file type: {uploaded_file.name}")
            return JsonResponse({'error': 'Unsupported file type. Only images (jpg, png, etc.) are allowed.'}, status=400)

        # 고유한 파일 이름 생성 (보안 및 중복 방지)
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        
        # 저장될 파일의 전체 경로 (MEDIA_ROOT/uploads/고유파일이름.확장자)
        upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
        os.makedirs(upload_dir, exist_ok=True) # 'uploads' 디렉토리가 없으면 생성

        file_path = os.path.join(upload_dir, unique_filename)

        try:
            # 파일 저장
            with open(file_path, 'wb+') as destination:
                for chunk in uploaded_file.chunks():
                    destination.write(chunk)
            
            # 클라이언트에게 반환할 상대 경로 (MEDIA_URL부터 시작하는 경로)
            # 예: 'uploads/abc-123.jpg'
            relative_path_for_client = os.path.join('uploads', unique_filename).replace('\\', '/')
            
            logger.info(f"Image uploaded successfully: {relative_path_for_client}")
            return JsonResponse({'message': 'Image uploaded successfully!', 'image_path': relative_path_for_client})

        except Exception as e:
            logger.exception(f"Error saving uploaded file {uploaded_file.name}: {e}")
            return JsonResponse({'error': f'Failed to save image: {str(e)}'}, status=500)
        
    logger.warning(f"Invalid request or no image provided for upload: {request.method}")
    return JsonResponse({'error': 'Invalid request or no image provided'}, status=400)


# ⭐ 이미지 검색 (img2img) 처리 뷰 (GET 요청) ⭐
def img2img_view(request, image_path):
    # image_path는 클라이언트로부터 받은 상대 경로 (예: 'uploads/abc-123.jpg')
    
    # 전체 이미지 파일 경로 구성
    full_image_path = os.path.join(settings.MEDIA_ROOT, image_path)

    # 1. 파일 존재 여부 확인
    if not os.path.exists(full_image_path):
        logger.error(f"Image not found at path: {full_image_path}")
        return JsonResponse({'error': 'Image not found'}, status=404)
    
    # 2. 파일이 유효한 이미지인지 확인 (MIME 타입 또는 확장자 기반)
    allowed_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
    file_extension = os.path.splitext(full_image_path)[1].lower()
    if file_extension not in allowed_extensions:
        logger.error(f"Requested file is not an image: {full_image_path}")
        return JsonResponse({'error': 'Requested file is not a valid image.'}, status=400)


    # ⭐ 3. 여기에 실제 이미지 처리 (img2img) 로직을 구현합니다 ⭐
    # image_path를 사용하여 Pillow (PIL) 등으로 이미지를 열고, 원하는 변환/분석 작업을 수행합니다.
    # 예시: 이미지 파일을 열고 간단한 정보 추출
    try:
        with Image.open(full_image_path) as img:
            width, height = img.size
            image_format = img.format
            mode = img.mode
            
            processed_result = {
                "requested_image": image_path,
                "message": "Image processed successfully (dummy data)",
                "image_details": {
                    "filename": os.path.basename(full_image_path),
                    "width": width,
                    "height": height,
                    "format": image_format,
                    "mode": mode
                },
                # "generated_image_url": "http://127.0.0.1:8000/media/generated_images/new_image.jpg", # 생성된 이미지가 있다면 URL 포함
                "analysis_result": "This is where your AI/ML img2img output goes."
            }
        logger.info(f"Image processed for img2img: {image_path}")
        return JsonResponse(processed_result)

    except FileNotFoundError:
        logger.error(f"Image not found during processing: {full_image_path}")
        return JsonResponse({'error': 'Image file not found for processing.'}, status=404)
    except Exception as e:
        logger.exception(f"Error processing image {full_image_path}: {e}")
        return JsonResponse({'error': f'Image processing failed: {str(e)}'}, status=500)