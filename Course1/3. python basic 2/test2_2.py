# list comprehension

result = [i for i in range(10)]

print(result)

case_1 = ["A", "B", "C"]
case_2 = ["D", "E", "A"]

result = [i+j for i in case_1 for j in case_2]

print(result)

# enumerate, zip

alist = ['a1', 'a2', 'a3']
blist = ['b1', 'b2', 'b3']

for i, (a,b) in enumerate(zip(alist,blist)):
    print(i, a, b)