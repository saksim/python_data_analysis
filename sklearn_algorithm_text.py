#! -*-encoding=utf-8-*-
import numpy as np
n_topics = 10
n_topic_words=10
n_features = 1000
corpus = [ ]
import pandas as pd
'''导入数据'''
for line in open('F:\\WeiWeiHe\\data.txt'):
    corpus.append(line)
print corpus
''' 经过验证，是错误的做法。【转换为字符串型的分割单词，而需要输入的是文章形式】
导入数据
for line in open('F:\\WeiWeiHe\\data.txt'):
    corpuss.append(line.split(' '))
处理为strings数据
for i in range(0,corpuss.__len__()):
    corpuss[i] = str(corpuss[i])
corpuss = np.array(corpuss).ravel()
corpus = ""
corpus = corpus.join(corpuss)
print corpus
'''
def print_topic_words(model,feature_names,n_topic_words):
    for topic_idx,topic in enumerate(model.components_):
        print ("Topic # %d:" %topic_idx)
        print (" ".join([feature_names[i]
                         for i in topic.argsort()[:-n_topic_words - 1:-1]]))

'''使用tf-idf'''
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vector = TfidfVectorizer(max_df=0.95,min_df=2,max_features=n_features)
tfidf = tfidf_vector.fit_transform(corpus)
print tfidf.shape
print tfidf
'''使用tf特性做隐式狄里克雷聚合'''
from sklearn.feature_extraction.text import CountVectorizer
tf_vector = CountVectorizer(max_df=0.95,min_df=2,max_features=n_features)
tf = tf_vector.fit_transform(corpus)
print tf.shape
'''拟合NMF模式'''
from sklearn.decomposition import NMF
nmf = NMF(n_components=n_topics,random_state=1,alpha=0.1,l1_ratio=0.5).fit(tfidf)

'''TOPICS'''
tfidf_feature_ames = tfidf_vector.get_feature_names()
print_topic_words(nmf,tfidf_feature_ames,n_topic_words)

'''使用LDA算法尝试'''
from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_topics=n_topics,max_iter=5,learning_method='online',learning_offset=50.,random_state=0)
lda.fit(tf)
tf_feature_names = tf_vector.get_feature_names()
print_topic_words(lda,tf_feature_names,n_topic_words)
