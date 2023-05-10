import os

imgfile = "./image/image3.png"

name, ext = os.path.splitext(imgfile)
tarfile = os.path.join(name+"pred"+ext)
print(tarfile)
