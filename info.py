import time

version = "1.0.0"
MathY = "MathY "+version
mathy = "You're spelling it wrong man, MathY is the right way to go!"

print("Welcome to MathY--A Simple Tool For Solving Math Problems")
print("Version",version)
print("David L. Xu")
print()
print("MathY is a simple math toolbox made from sratch using python " \
      "language properties. It is for educational or recreational purpose only. ")
print("Additional information about MathY is available at https://github.com/DavidLXu/MathY.")
print("Please contribute if you find this software useful.")
print()

print("Type quicklook() to quickly look up the functions by group.")
localtime = time.asctime( time.localtime(time.time()) )
print (localtime)

def quicklook():
    print("Type dir() to see all functions available.")
    print("Type readme() to see the detailed documentation.")
    option = input("1. Basics\n2. Calculs\n3. Linear Algebra\n4. Numeric\nchoose:")
    print("not yet implemented, wait for the next version")
    if option == "1":
        pass
    elif option == "2":
        pass

def readme():
    with open(r'README.md', encoding="utf-8") as file:
        print(file.read())
