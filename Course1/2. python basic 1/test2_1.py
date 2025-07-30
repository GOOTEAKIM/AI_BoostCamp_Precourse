# 함수

def calculate_rectangle_area(x,y) :
    return x*y

rectangle_x = 10
rectangle_y = 20

print(rectangle_x, rectangle_y)

print("넓이 :", calculate_rectangle_area(rectangle_x, rectangle_y))

# 함수 vs 함수

def f(x): 
    return 2 * x +7


def g(x):
    return x**2


x = 2

print(f(x) + g(x) + f(g(x)) + g(f(x)))