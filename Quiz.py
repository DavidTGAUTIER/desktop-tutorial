DICTIONARY = {
    ("what is the name for the fear of animals ? : ", "zoophobia "),
    ("Where is mount rushmore ? : ", "dakota"),
    ("When was the reunification of vietnam ? : ", "1975")
}

class Quiz:
    
    question, answer = 0, 1
    
    def __init__(self, dictionary):
        self.dictionary = dictionary
        
    def start(self):
        for number, content in enumerate(self.dictionary, start=1):
            print(f"Question {number} : {content[Quiz.question]}")
            
            nb_try = 3
            while nb_try:
                answer = input().lower()
                print(f"Your answer : {answer}")
                if answer != content[Quiz.answer]:
                    print("too bad.. That's not the correct answer")
                    nb_try -= 1
                    print(f"Sorry you have {nb_try} left")
                else:
                    print("Good job! This is the right answer")
                    break
            
            if not nb_try:
                print("Too bad, you lost the game..")
                break
            
        print("You win the game !!!")

if __name__ == '__main__':
    quiz = Quiz(dictionary = DICTIONARY)
    quiz.start()