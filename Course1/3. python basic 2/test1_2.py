# set

s = set([1,2,3,1,2,3])

print(s)

s.add(1)
print(s)

s.remove(1)
print(s)

s.update([1,6,3,7])
print(s)

s.discard(3)
print(s)

s.clear()
print(s)

s1 = set({1,2,3,4,5})
s2 = set({3,4,5,6,7})

print(s1|s2)

print(s1-s2)

print(s2-s1)

print(s1&s2)
