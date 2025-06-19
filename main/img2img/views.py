from django.shortcuts import render

# Create your views here.
def img2img2_view(request):
    return render(request,"img2img.html")

def result(request):
    return render(request,"img2img_result.html")