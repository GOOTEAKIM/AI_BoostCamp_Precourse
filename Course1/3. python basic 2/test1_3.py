country_code = {}

country_code = {"america" : 1, "korea" : 2, "china" : 3, "japan" : 4}

print(country_code)

print(country_code.keys())
print(country_code.values())

print(country_code["america"])

country_code["hong kong"] = 5

print(country_code)

for k,v in country_code.items() :
    print("key :", k)
    print("value :", v)