import cv2
import numpy as np

img = cv2.imread("./rrr/11.png", 1)  # 读取图像，参数“1”等价于cv2.IMREAD_COLOR，将图像调整为3通道的BGR图像，该值是默认值
w, h, c = img.shape  # 获取图像的行数、列数、通道数
mask = cv2.imread('./rrr/12.png')  # 创建掩模图像，(w,h,c)可直接写为img.shape
#mask[170:220, 110:160] = 255  # 保留ROI
x = cv2.add( img,mask)  # 将原始图像和掩模图像进行按位与运算
#x = cv2.bitwise_or(img,mask)
# 打印原始图像和掩模图像的属性
print("img.shape=", img.shape)
print("mask.shape=", mask.shape)
# 显示图像
# cv2.imshow("original image", img)
# cv2.imshow("mask", mask)

cv2.imshow("cv2.bitwise_xor", x)
cv2.imwrite("./rrr/yuan1mask_pred.jpg",x)
#
# cv2.bitwise_and()	按位与
# 2	cv2.bitwise_or()	按位或
# 3	cv2.bitwise_xor()	按位异或
# 4	cv2.bitwise_not()
# #
# ----------释放窗口---------
cv2.waitKey()
cv2.destroyAllWindows()


# 1.30 图像的叠加