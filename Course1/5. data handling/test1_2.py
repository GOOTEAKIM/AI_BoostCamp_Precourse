while True:
    value = input()

    for digit in value:
        if digit not in "0123456789":
            raise ValueError
    
    print("정수로 변환된 숫자", int(value))