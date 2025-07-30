def make_grade(x):

    if x >= 95 :
        print("A+")
    elif x < 60:
        print("F")
    else :
        print("C")
    
score = int(input())
make_grade(score)
