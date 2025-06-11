from datetime import date
# yyyymmdd
def date2int(day:str):
    day = str(day)
    try:
        y = int(day[:4] )
        m = int(day[4:6])
        d = int(day[6:] )
    except:
        return 0
    #2022-2-22 = 1
    first = date(2022,2,21)
    now   = date(y,m,d)
    return int(str(now-first).split()[0])

# 릴리 해원 설윤 배이 지우 규진 기타 동물
people = ["ㄷㅊ","ㄹㄹ", "ㅎㅇ", "ㅅㅇ", "ㅂㅇ", "ㅈㅇ", "ㄱㅈ", "ㄱㅌ", "ㄷㅁ"]
# 퀄리파잉 o.o 다이스 펑글크 럽미 파티어클락 대쉬 별별별 노어밧미
album = ["ㅋㄿㅇ","ㅇㅇ","ㄷㅇㅅ","ㅍㄱㅋ","ㄹㅁ","ㅍㅌㅇㅋㄹ","ㄷㅅ","ㅂㅂㅂ","ㄴㅇㅂㅁ"]
# 분류
sort = ["ㅁㅂ","음방","콘서트","홈마","cover","vlog","live","비하인드","HBD"]
# 자컨
contents = ["입덕투어","이슈클럽","회포자","워크돌","설중","절전동","그림일기"
            ,"챗톡","blog","이상한","차개듀","쮸뀨미"]
# 어린이 앞머리 장발 자막 
element = ["어린이", "앞", "단", "금", "cc"]