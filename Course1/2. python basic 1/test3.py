# indexing
colors = ['red', 'blue', 'green']

print(colors)
print(colors[0])
print(len(colors))

# slicing

cities = ['서울', '대구', '부산', '울산', '대전']

print(cities[:])
print(cities[::-1])
print(cities[::2])

# 리스트의 연산

color = ['red', 'blue', 'green']
color2 = ['orange', 'black', 'white']

print(color + color2)

color[0] = 'yellow'

print(color*2) # 리스트 반복

print('blue' in color) # 특정 문자열이 리스트에 있는지 확인
print('blue' in color2)

# 추가, 삭제 연산
color.append("white")
print(color)

color.extend(["black", "purple"])
print(color)

color.insert(0, "orange")
print(color)

color.remove("white")
print(color)

del color[0]
print(color)

# 패킹, 언패킹

t = [1,2,3]

a,b,c = t

print(t, a,b,c)

# 2차원 리스트

kor = [49,79,20,100,80]
math = [43,59,85,30,90]
eng = [49,79,48,60,100]

score = [kor,math,eng]

print(score)
print(score[0])
print(score[1][2])