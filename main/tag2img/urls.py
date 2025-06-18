from django.urls import path
from . import views

urlpatterns = [
    path('', views.tag2img_view, name='tag2img'),
    path('input/', views.input_form_view, name='input_form'),
    path('result/', views.result_display_view, name='result_display'),
]
