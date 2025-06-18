from django.urls import path
from . import views

urlpatterns = [
    path('', views.img2img2_view, name='img2img2'),
    path('result/', views.result, name='result'),
]
