# 要提取图像中的边缘，可以使用Python的OpenCV库。OpenCV提供了许多图像处理函数和算法，其中包括边缘检测。
#
# 以下是一个简单的示例：
#
# ```python
import PIL.Image
import cv2
import numpy as np
from PIL import Image


# 读取图像
img = cv2.imread('image\\1mask_pred.png')
img = np.array(Image.fromarray(img).resize([256,256], resample=PIL.Image.BILINEAR))
# 转换为灰度图像
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 进行边缘检测
edges = cv2.Canny(gray, 256, 256)

# 显示图像
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
# ```
#
# 这段代码会读取名为`image.jpg`的图像文件，将其转换为灰度图像，然后使用Canny算法进行边缘检测。最后，将提取出的边缘图像显示出来。
#
# 在Canny函数中，第二个和第三个参数分别表示低阈值和高阈值。这些值可以根据需要进行调整。如果低阈值和高阈值之间的梯度值高于高阈值，则该像素被视为边缘像素。如果低阈值和高阈值之间的梯度值低于低阈值，则该像素被视为非边缘像素。如果梯度值在低阈值和高阈值之间，则只有当该像素与边缘像素相邻时，才被视为边缘像素。
