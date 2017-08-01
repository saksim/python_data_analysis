# -*-encoding=utf-8-*-
import numpy as np
import pandas as pd
'''
df = pd.read_table('F:\\data\\hly-wind-1stdir2.txt',sep=',', header = None)'''
'''重要的基本函数
（1）HEAD和TAIL'''
long_series = pd.Series(np.random.rand(1000))
print long_series.head()
print long_series.tail()

'''
(2)包含元素和原始ndarray()
print df.values
'''
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
4.2) df['列名']抽取的是一列切片数据
column = df['jcc']
print column
'''
'''
4.3) .sub [Subtraction 即：两个数据之间的差异]
print df.sub(row,axis='columns')
print df.sub(column,axis = 'index')
'''
'''注意：如果是直接使用NP，则必须是等长度的！'''
'''df = pd.DataFrame({'HS':np.random.rand(10),'MJ':np.random.random(5),'CJ':np.random.randn(9)})'''
df = pd.DataFrame({'HS':pd.Series(8000*np.random.rand(10), index= ['heweiwei0107','heweiwei01071','wh13624','qq2994146706','heweiwie01072','heweiwei0109','cstdm','9692309982','heweiwei01073','249146147']),
                   'MJ':pd.Series(5000*np.random.random(5), index= ['heweiwei01071','wh13624','cstdm','9692309982','249146147']),
                   'CJ':pd.Series(10000*np.random.rand(9), index= ['heweiwei0107','heweiwei01071','wh13624','qq2994146706','heweiwie01072','cstdm','9692309982','heweiwei01073','249146147'])})
print df

'''
4.4)使用pandas序列设置多重索引DF
'''
dfmi = df.copy()
dfmi.index = pd.MultiIndex.from_tuples([('heweiwei','1'),('heweiwei','2'),('wh13624','1'),('qq','1'),('heweiwei','3'),
                                        ('heweiwei','4'),('heweiwei','5'),('relation','1'),('relation','2'),('heweiwei','6')],
                                       names=['first','second'])
column = dfmi['HS']
dfmi = dfmi.sub(column,axis=0,level='second')
print dfmi

'''
4.5)groupby 对所选列，按照列值，对剩余列的值进行分组
'''
dfmi = df.copy()
dfmi = dfmi.groupby(by=['CJ','HS']).sum()
print dfmi

'''20170727
计划：
1)选择随机样本P585
2)扩大化设置 P587
3)快速规模化设置
4）布尔值索引
5）ISIN索引
6）WHERE方法以及标记
7）经典query()方法
8）多重索引query()语法
9）query()使用案例
10）in和not in操作符 P602
'''

'''20170801'''
dfmi = df.copy()
'''where条件:选择满足条件的DF'''
'''1)'''
hww = dfmi.where(dfmi>5000.0)
print hww
'''2)'''
hww1 = dfmi[dfmi > 5000.0]
print hww1
'''3)使用OTHER参数，用于替代条件判断为假时的值'''
hww2 = dfmi.where(dfmi >5000.0, dfmi*100)
print hww2

'''4)输入布尔值条件进行部分选择，通过.ix返回值'''
dfmi[ dfmi[1:4] > 5000] = 3
print dfmi

'''5)对轴和标签常量结合WHERE分配输入【即：保留满足条件的数据，不满足条件的数据按照index所在的df[列]数据进行代替】'''
hww3 = dfmi.where(dfmi>5000.0,dfmi['CJ'],axis = 'index')
print hww3
df = pd.DataFrame([[1,2,3],[4,5,6],[7,8,9]])
df = df.rename(columns={0:'a',1:'b',2:'c'})
print df
hww3 = df.where(df > 5, df['b'], axis='index')
print hww3

'''6)标记mask【将满足条件的数据设置为缺失值】'''
hww4 = df.mask(df>=5)
print hww4

'''query()方法'''
n = 10
df = pd.DataFrame(np.random.rand(n,3),columns=list('abc'))
'''python所选'''
print df
hww1 = df[(df.a < df.b) & (df.b < df.c)]
print hww1
'''query'''
hww1 = df.query('(a<b) & (b<c)')
print hww1
df = pd.DataFrame(np.random.randint(n / 2, size=(n, 2)), columns= list('bc'))
'''2)不使用某列作为索引，而是直接用索引'''
hww = df.query('index < b < c')
print hww
'''1)若想将某列作为索引，同时完成比较'''
df.index.name = 'a'
print df
hww = df.query('a<b and b <c')
print hww

'''3)如果索引名与列名重复，则选择时，这个列名优先'''
df = pd.DataFrame({'b':np.random.randint(n,size=n / 2)})
df.index.name = 'b'
print df
hww = df.query('b>2.5')
print hww
'''3))需要使用明确指定index索引，才能对索引进行操作'''
hww = df.query('index > 2.5')
print hww

'''20170802 多重索引进行QUERY查询语法'''

