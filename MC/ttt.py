def aaa():
    if not hasattr(aaa, "a"):
        aaa.a = 0
    aaa.a+=1
    print(aaa.a)

aaa()