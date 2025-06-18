from django.shortcuts import render,redirect
from django.urls import reverse

def tag2img_view(request):
    return render(request, 'tag2img.html')

def input_form_view(request):
    return render(request, 'input_form.html')

def result_display_view(request):
    if request.method == 'POST':
        selected_person = request.POST.getlist('person')
        selected_activity_period = request.POST.getlist('activity_period')
        selected_category = request.POST.getlist('category')
        selected_content_name = request.POST.getlist('content_name')
        selected_element = request.POST.getlist('element')

        context = []

        text2two = ""
        for j in ["단체","릴리", "해원", "설윤", "배이", "지우", "규진","동물","기타"]:
            two2ten += "1" if j in selected_person else "0"
        context.append(int(two2ten,2))

        text2two = ""
        for j in ["QUALIFYING", "AD MARE", "ENTWURF", "expérgo", "AMND", "Fe3O4: BREAK", "Fe3O4: STICK OUT", "Fe3O4: FORWARD"]:
            two2ten += "1" if j in selected_person else "0"
        context.append(int(two2ten,2))

        text2two = ""
        for j in ["컨셉포토","포토북","음방","콘서트","홈마","live","sns"]:
            two2ten += "1" if j in selected_person else "0"
        context.append(int(two2ten,2))

        text2two = ""
        for j in ["입덕투어","이슈클럽","회포자","워크돌","설윤중심","절전동",그림일기 챗톡 vlog 이상한나라의엔믹스 차캐듀 비하인드 cover HBD]:
            two2ten += "1" if j in selected_person else "0"
        context.append(int(two2ten,2))


        return render(request, 'tag2img_result.html', context)
    else:
        return redirect('input_form')