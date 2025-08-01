class SoccerPlayer(object) :
    def __init__(self, name, position, back_number):

        self.name= name
        self.position = position
        self.back_number = back_number

    def change_back_number(self, new_number):
        
        print("등번호 변경")
        
        self.back_number = new_number

gootea = SoccerPlayer("gootea", "DF", 4)

print(gootea.back_number)
print(gootea.position)



        