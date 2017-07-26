# -*-encoding=utf-8-*-
import numpy as np
import pandas as pd

df = pd.read_table('F:\\data\\hly-wind-1stdir2.txt',sep=',', header = None)
'''重要的基本函数
（1）HEAD和TAIL'''
long_series = pd.Series(np.random.rand(1000))
print long_series.head()
print long_series.tail()

'''
(2)包含元素和原始ndarray()'''
print df.values

'''
(3)加速操作 Accelerated Operation
pandas通过numexpr支持加速确定类型的二进制数字和布尔值算子 【提高了> * +等操作的速度】'''

'''
（4）匹配/广播行为'''
df = pd.DataFrame({'hww':pd.Series(np.random.rand(3),index=['DPS','HPS','SPS']),
                   'lyl': pd.Series(np.random.rand(4), index=['DPS', 'HPS', 'SPS','DMG']),
                   'jcc':pd.Series(np.random.rand(3),index=['DMG','HPS','SPS'])})
print df
'''
4.1) .ix 代表的是行切片数据'''
row = df.ix[1]
print row
print df.ix[1:3]

'''
4.2) df['列名']抽取的是一列切片数据'''
column = df['jcc']
print column
'''
4.3) .sub [Subtraction 即：两个数据之间的差异]'''
print df.sub(row,axis='columns')
print df.sub(column,axis = 'index')

'''P490'''
