#! -*-encoding=utf-8-*-
import pandas as pd
import hypertools as hyp
from hypertools.tools import cluster
data = pd.read_csv('F:\\mushrooms.csv')
#print data.head()

'''
Now let’s plot the high-dimensional data in a low dimensional space by passing it to HyperTools.
To handle text columns, HyperTools will first convert each text column into
    a series of binary ‘dummy’ variables before performing the dimensionality reduction.
For example, if the ‘cap size’ column contained ‘big’ and ‘small’ labels,
    this single column would be turned into two binary columns:
        one for ‘big’ and one for ‘small’, where 1s represents the presence of that feature and 0s represents the absence
            (for more on this, see the documentation for the get_dummies function in pandas).
'''
hyp.plot(data,'o') #高维度数据在低纬度空间中的展示图（使用HyperTools）

#由以上所画图可知，相似的特征出现在近邻的群中，并且很明显有几个不一样的特战群。
#即：所有的特征群不一定是完全相等，之后可以根据数据中我们喜欢的特征进行颜色标记。

#hyp.plot(data,'o',group=class_labels.legend=lisk(set(class_labels)))
#以上需要预先分类，才能使用

hyp.plot(data,'o',n_clusters=50)#根据分类数进行染色展示

#为了得到登录的群标签，聚类工具可能会通过hyp.tools.cluster被直接调用,并得到类别结果再传递给plot
#[注意：对于母包和子包，如果导入母包没有用*，则无法识别子包!!!]
cluster_labels = cluster(data,n_clusters=50)
hyp.plot(data,'o',group=cluster_labels)

#[注意：HYPERTOOLS默认使用PCA进行降维，所以如果想用其他的算法进行降维，可以如下]
from sklearn.manifold import TSNE
from hypertools.tools import df2mat
TSNE_model = TSNE(n_components=3)
reduced_data_t = TSNE_model.fit_transform(df2mat(data))
hyp.plot(reduced_data_t,'o')