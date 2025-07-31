# stack

word = input()

word_list = list(word)

for i in range(len(word_list)):
    print(word_list.pop())

# queue

a = [1,2,3,4,5]

a.append(10)
a.append(20)

first = a.pop(0)
print(first)

a.pop(0)
print(a[0])

# tuple

t = (1,2,3)

print(t+t, t*2)

print(len(t))

# t[1] = 5 # 에러 발생, 값의 변경 불가


