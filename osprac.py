import os
path = "C:\\yolo"

retval = os.getcwd()
print(retval)
os.chdir( path )
print(os.getcwd())
os.chdir(os.path.join("fake", "annotation"))
print(os.getcwd())
