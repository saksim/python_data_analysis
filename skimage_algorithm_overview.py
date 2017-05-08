#! -*-encoding=utf-8-*-
from skimage import data
camera = data.camera()
print type(camera) #显示其类型，发现这些数据为mumpy格式类型

print camera.shape #查看数据形状
print camera.size #查看总大小

print camera.min()
print camera.max()
