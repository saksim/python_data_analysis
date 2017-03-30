#!-*- coding=utf-8 -*-
'''
使用pandas做训练
PANDAS有两个重要的东西：SERIES和DATAFRAME
'''
import numpy as np
import pandas as pd

'''
查看输出Series的形式
'''
obs = pd.Series([4,7,5,-3])
print "Series数据的形式为:"
print obs
print "***********************************************"
'''
由以上可知：Series是由index和values组成
分别查看
'''
print "Series的索引为："
print obs.index
print "Series的数值为："
print obs.values
print "***********************************************"
'''对各个数据点进行标记的索引'''
obs2 = pd.Series([4,7,5,-3],index=['he','li','liu','she'])
print "带标记索引的数值点："
print obs2
print "带标记索引series的索引："
print obs2.index
print "***********************************************"
'''特点：1）可以通过索引的方式选取Series中的单一或一组值'''
print obs2["he"]
print "***********************************************"
'''特点2） 可以运算'''
print obs2[obs2 > 1.0] **2
print np.log10(obs2[obs2 > 1.0] **2)
print "***********************************************"
'''特点3） 可以将Series看做定长的有序字典'''
print 'he' in obs2
print 'weiwei' in obs2
print "***********************************************"
'''特点4） 存于字典中的数据可以创建Series'''
sdata = {"heweiwei":13624,"liuyifei":13721,"zhaoxin":14928}
obs3 = pd.Series(sdata)
print obs3
print "***********************************************"
'''特点5）Series具有name属性，很关键'''
obs3.index.name = "CANDIDATE_NUMBER"
obs3.name = "CANDADATE_INFO"
print obs3
print "***********************************************"

'''PANDAS的第二个主要数据结构DATAFRAME'''
data = {
    "state": ["China","Japan","American","Canada"],
    "year": [2013,2015,2014,2016],
    "pop": [13.0,2.9,3.2,2.9]}
frame = pd.DataFrame(data)
print frame
print "***********************************************"

'''DataFrame指定列序列'''
frame1 = pd.DataFrame(data,columns=['year','pop','state'])
print frame1
print "***********************************************"
'''传入列在数据中找不到，则产生NA值'''
frame2 = pd.DataFrame(data,columns=['year','pop','state','debt'],index=['one','two','three','four'])
print frame2
print "***********************************************"
'''转置'''
print frame2.T
print "***********************************************"

'''PANDAS中的索引对象
负责管理轴标签和其他元数据【构建Series和DataFrame时，用的任何数组或其他序列的标签都会被转换成一个索引】
索引不可修改！这样在多结构间保证安全共享
'''

'''重新索引'''
obs4 = pd.Series([4.5,7.2,-5.3,3.6],index=['d','b','a','c'])
print obs4
obs4_re = obs4.reindex(['c','mayifei','yangxinyu','yaoyutong','heweiwei','god','saksim'])
print obs4_re
print "***********************************************"
'''使用ffill前向值填充'''
obs5 = pd.Series(["hww","MYF","YxY","yyt"],index=[0,1,3,7])
print obs5.reindex(range(10),method='ffill')
print "***********************************************"
'''使用bfill后向值填充'''
obs6 = pd.Series(["hww","MYF","YxY","yyt"],index=[0,1,3,7])
print obs6.reindex(range(10),method='bfill')
print "***********************************************"
'''丢弃指定轴上的项
可以删除任意轴的索引值
'''
hww = pd.Series(np.arange(5.),index=['a','b','c','d','e'])
hww_drop = hww.drop('c')
print hww_drop
print "***********************************************"
'''填充数据'''
df1 = pd.DataFrame(np.arange(20.).reshape((5,4)),columns=['hww','lyl','lyf','myf'])
print df1
'''运算时必须一样大小，否则会有NA值'''
df2 = pd.DataFrame(np.arange(20.).reshape((4,5)),columns=['hww','lyl','lyf','myf','saksim'])
print df2
print np.log10(df1 ** df2)

'''对数据进行相关处理运算'''
'''通过NUMPY导入数据到PANDAS中'''
all_data_a = np.loadtxt("F:\\WeiWeiHe\\beh_master\\spark-1.6.3-bin-hadoop2.6\\data\\mllib\\lr_data.txt")
print all_data_a.shape
all_data_b = pd.DataFrame(all_data_a, columns=['x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11'])
'''print all_data_b'''

'''相关系数corr()'''
xgxs = all_data_b.corr()
print xgxs

'''协方差矩阵cov()'''
xfcjz = all_data_b.cov()
print xfcjz

'''某一行或列与另一Series或者DataFrame之间的相关系数 corrwith?????????如何实现？因为列和行无法定位'''

'''PANDAS重要功能：层次化索引
在一个轴上拥有多个索引级别，能以低维度形式处理高纬度数据
每个轴都可以分层索引
'''
all_data_1 = pd.Series(np.random.randn(10),
                       index=[['a','a','a','b','b','b','c','c','d','d'],[1,2,3,1,2,3,1,2,2,3]])
print all_data_1
print all_data_1.index
print "***********************************************"

'''层次化索引的操作：选子集、内层选取
用途：数据重塑
    基于分组的操作
'''
print all_data_1['b':'c']
print all_data_1[:,2]

print all_data_1.unstack()
