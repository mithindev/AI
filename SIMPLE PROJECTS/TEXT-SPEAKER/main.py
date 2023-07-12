import pyttsx3

if __name__ == '__main__':
    print("Welcome!")
    while True:
        x = input("Enter what do you want me to pronounce :")
        if x == "exit":
            break
        engine = pyttsx3.init()
        engine.say(x)
        engine.runAndWait()