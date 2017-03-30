#! -*- encoding=utf-8 -*-
''' 文本格式数据的导入
read_csv 加载带分隔符的数据，默认分隔符为逗号
read_table 读取默认分隔符为制表符(/t)的数据
read_fwf 读取定宽列格式数据【即：无分隔符】
read_clipboard 读剪切板中的数据【多用于网页转换为表格】
'''

import pandas as pd

'''读CSV文件'''
df = pd.read_csv("F:/WeiWeiHe/beh_master/spark-1.6.3-bin-hadoop2.6/data/mllib/sample_tree_data.csv")

'''使用read_table度csv'''
df2 = pd.read_table("F:/WeiWeiHe/beh_master/spark-1.6.3-bin-hadoop2.6/data/mllib/sample_tree_data.csv",sep=',')

'''处理无标题的方案：添加参数header=none,或者自己添加标题'''
'''df = pd.read_csv("F:/WeiWeiHe/beh_master/spark-1.6.3-bin-hadoop2.6/data/mllib/sample_tree_data.csv",header=None)'''
'''print df'''

df = pd.read_csv("F:/WeiWeiHe/beh_master/spark-1.6.3-bin-hadoop2.6/data/mllib/sample_tree_data.csv",names=['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','aa','bb','cc','dd'])
print df
'''也可以为这个DataFrame做个索引'''
'''
name = ['X','XX','XXX','……','XXXXXX','XXXX','message']
pd.read_csv('xxxx.csv',names=name,index_col='message')
'''

'''print df_new'''

'''frame = pd.DataFrame(df,columns=list('abcdefghijklmnopqrstuvwxyz'))'''
