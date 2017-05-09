#! -*-encoding=utf-8-*-

#1)例子展示
from skimage import img_as_float
import numpy as np
image = np.arange(0,50,10,dtype=np.uint8)
print image.astype(np.float),img_as_float(image)

#2.1)输入数据类型
from skimage import img_as_ubyte
image = np.array([0,0.5,1],dtype=float)
print img_as_ubyte(image)

#2.2)本地忽略因为转换类型导致准确度损失而带来的警告
import warnings
from skimage import img_as_ubyte
image = np.array([0,0.5,1],dtype=float)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    img_as_ubyte(image)
print img_as_ubyte(image)

#2.3)PRESERVE_RANGE参数，会对具有物理度量的图片，保留其取值范围【默认时是false不保持】
from skimage import data
from skimage.transform import rescale
import numpy as np
image = data.coins()
print image.dtype,image.min(),image.max(),image.shape
rescaled = rescale(image,0.5)
#不保存之前
print (rescaled.dtype,np.round(rescaled.min(),4),
 np.round(rescaled.max(),4),rescaled.shape)
#保存之前
rescaled = rescale(image,0.5,preserve_range=True)
print (rescaled.dtype,np.round(rescaled.min(),4),
 np.round(rescaled.max(),4),rescaled.shape)

#3)输出格式
from skimage import data
from skimage import img_as_uint
from skimage import filters
import matplotlib.pyplot as plt
image = data.coins()
plt.imshow(image)
plt.show()
out = img_as_uint((filters.sobel(image)))
plt.imshow(out)
plt.show()


#4)可以与OpenCV进行耦合使用
#5)