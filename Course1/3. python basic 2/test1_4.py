from collections import deque

deque_list = deque()

for i in range(5):
    deque_list.append(i)

deque_list.appendleft(10)
deque_list.append(100)

print(deque_list)

deque_list.rotate(1) # 시계 방향으로 1칸
print(deque_list)

deque_list.extend([5,6,7])
print(deque_list)
