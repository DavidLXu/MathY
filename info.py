import time

version = "1.0.0"


print("Welcome to MathY--A Simple Tool For Solving Math Problems")
print("Version",version)
print("David L. Xu")
print()
print("MathY is a simple math toolbox made from sratch using python \
    language properties. It is for educational or recreational purpose only. ")
print("Additional information about MathY is available at https://github.com/DavidLXu/MathY.")
print("Please contribute if you find this software useful.")
print()
print("Use dir() to see all functions available.")

localtime = time.asctime( time.localtime(time.time()) )
print (localtime)