for city in ["seoul", "busan", "ulsan"]:
    print(city, end="\t")

for char in "Python is easy":
    print(char, end=" ")


cities = ["seoul", "busan", "ulsan"]

iter_obj = iter(cities)

print(iter_obj)

print(next(iter_obj))
print(next(iter_obj))
print(next(iter_obj))
# next(iter_obj)
