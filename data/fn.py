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

print(date2int(20220222))