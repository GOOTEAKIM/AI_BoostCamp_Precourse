def factorial(n):
    if n == 1:
        return 1
    else:
        return n * factorial(n-1)
    
print(factorial(int(input())))

n = int(input())

ans = 1

while n > 0:
    
    ans *= n

    n-=1

print(ans)