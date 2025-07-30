def test(t):
    print(x)
    t = 20
    print(t)

x = 10
test(x)
# print(t) # 에러 발생

def f():
    global s
    s = "I love Ulsan"
    print(s)

s = "I love Paris"
f()
print(s)

