#! -*-encoding=utf-8-*-
from skimage import data
import numpy as np
camera = data.camera()
#得到统计数据
print camera.min(),camera.max(),camera.mean()
#得到尺寸大小和类型
print camera.size,camera.shape,type(camera)

##Numpy索引
#可以用于寻找像素值和调整像素值
print camera[10,20]
camera[10,20] = 0
print camera[10,20]
#第一维度代表的是行，第二维度代表的是列

#对像素集进行修改
#1)切片：比如将前十行设置为黑色
camera[:10]= 0

#2）掩码数组(mask)：由一个正常数组和一个布尔数组组成。布尔数组中值为TRUE的元素表示正常数组中对应下标的值无效 false有效
mask = camera < 87
camera[mask] = 255 #将掩码数组中为TRUE的设置为白像素

#3）fancy索引
inds_r = np.arange(len(camera))
inds_c = 4 * inds_r % len(camera)
camera[inds_r,inds_c] = 0

#使用掩码数组，在选择像素集合到展示更多操作时，特别有用
#掩码数组作为任何布尔值数组，且可以作为相同形状的图片
nrows, ncols = camera.shape
row, col = np.ogrid[:nrows, :ncols]
cnt_row, cnt_col = nrows / 2, ncols / 2
outer_disk_mask = ((row - cnt_row)**2 + (col - cnt_col)**2 >
                   (nrows / 2)**2)
camera[outer_disk_mask] = 0
import matplotlib.pyplot as plt
plt.imshow(camera) #使用matplotlib中imshow()函数对一个numpy数组进行画图展示
#以上不知为何，无图片
##原因：没用使用plt.show()
plt.show()
cat = data.chelsea()
print type(cat),cat.shape
reddish = cat[:,:,0] > 160
cat[reddish] = [0, 255, 0]
plt.imshow(cat)
plt.show()#plt创建了一个绘图空间 然后绘图，最后需要用命令才能进行显示
