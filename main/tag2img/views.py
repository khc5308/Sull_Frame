from django.shortcuts import render,redirect
import json,os

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
        for j in ["단체","릴리", "해원", "설윤", "배이", "지우", "규진","기타","동물"]:
            text2two += "1" if j in selected_person else "0"
        context.append(int(text2two,2))

        text2two = ""
        for j in ["QUALIFYING", "AD MARE", "ENTWURF","Funky Glitter Christmas", "expérgo", "AMND", "Fe3O4: BREAK", "Fe3O4: STICK OUT", "Fe3O4: FORWARD"]:
            text2two += "1" if j in selected_activity_period else "0"
        context.append(int(text2two,2))

        text2two = ""
        for j in ["컨셉포토","음방","콘서트","홈마","cover","vlog","live","비하인드","HBD"]:
            text2two += "1" if j in selected_category else "0"
        context.append(int(text2two,2))

        text2two = ""
        for j in ["입덕투어","이슈클럽","회포자","워크돌","설중","절전동","그림일기","챗톡","blog","이상한","차개듀","쮸뀨미"]:
            text2two += "1" if j in selected_content_name else "0"
        context.append(int(text2two,2))

        text2two = ""
        for j in ["어린이날", "앞머리", "단발", "금발", "자막"]:
            text2two += "1" if j in selected_element else "0"
        context.append(int(text2two,2))

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'vec2name.json')) as f:
            data = json.load(f)
        if str(tuple(context)) in data:
            post = {"data" : data[str(tuple(context))]}
        else:
            post = {"data" : ""}
    
        return render(request, 'tag2img_result.html', post)
    else:
        return redirect('input_form')