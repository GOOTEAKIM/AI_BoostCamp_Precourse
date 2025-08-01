# 상속

class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

class Korean(Person):
    pass

first_korean = Korean("gootea", 99)

print(first_korean.name)
print(first_korean.age)

class Employee(Person):
    
    def __init__(self, name, age, gender, salary, hire_date):
        super().__init__(name, age)
        self.gender = gender
        self.salary = salary
        self.hire_date = hire_date

gootea = Employee(first_korean.name, first_korean.age, "male", 10000, "17th")

print(gootea.name)
print(gootea.salary)
print(gootea.hire_date)
