#! -*-encoding=utf-8-*-
import numpy as np
n_topics = 10
n_topics_words = 10
n_features = 1000
corpus = [ ]
corpus_y =[ ]
def print_topic_words(model,feature_names,n_topic_words):
    for topic_idx,topic in enumerate(model.components_):
        print ("Topic # %d:" %topic_idx)
        print (" ".join([feature_names[i]
                         for i in topic.argsort()[:-n_topic_words - 1:-1]]))

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB

'''读取数据'''
for line in open('F:\\WeiWeiHe\\data.txt'):
    corpus.append(line)
print corpus
'''特征数据
vectorizer = TfidfVectorizer(encoding='latin1')
X_train =vectorizer.fit_transform(corpus)'''
'''随机新建标签数据，并读入'''
'''
for line in open('F:\\WeiWeiHe\\data_y.txt'):
    corpus_y.append(line)
print corpus_y
'''

'''分类器
import pandas as pd
import matplotlib.pyplot as plt
y_train = pd.read_table('F:\\WeiWeiHe\\data_y.txt',header=None)
y_train = np.array(y_train).ravel()
print y_train
X_test = X_train
Y_test = y_train
def benchmark(clf_class,params,name):
    print ("paramaters:",params)
    clf = clf_class(**params).fit(X_train,y_train)

    if hasattr(clf,'coer_'):
        print ("Percentage of non zeros coef: %f"
               % (np.mean(clf.coef_ != 0) * 100))
    print ("Predicting")
    pred = clf.predict(X_test)
    print (classification_report(Y_test,pred))
    cm = confusion_matrix(Y_test,pred)
    print ("Confusion Matrix:")
    print cm
    plt.matshow(cm)
    plt.title('%s' % name + u"分类器的混沌矩阵")
    plt.colorbar()

parameters = {
'loss': 'hinge',
'penalty': 'l2',
'n_iter': 50,
'alpha': 0.00001,
'fit_intercept': True,
}

benchmark(SGDClassifier,parameters,'SGD')
plt.show()

parameters = {'alpha': 0.01}
benchmark(MultinomialNB,parameters,'MultinomialNB')
plt.show()
'''

import pandas as pd
import matplotlib.pyplot as plt
y_train = pd.read_table('F:\\WeiWeiHe\\data_y.txt',header=None)
y_train = np.array(y_train).ravel()
X_train = corpus
'''获取Y中标签的类别总数'''
true_k = np.unique(y_train).shape[0]
print y_train
X_test = X_train
Y_test = y_train


from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer,TfidfTransformer
from sklearn.pipeline import make_pipeline
hasher = HashingVectorizer(n_features,non_negative=True,binary=False)
'''根据实际情况可能选择多种不同参数，
1）构建多从选择参数'''
from optparse import OptionParser
import sys
op = OptionParser()
op.add_option("--lsa",dest="n_components", type="int",help="Preprocess documents with latent semantic analysis.")
op.add_option("--no-minibatch",action="store_false", dest="minibatch", default=True,
              help="Use ordinary k-means algorithm (in batch mode).")
op.add_option("--no-idf",action="store_false", dest="use_idf", default=True,
              help="Disable Inverse Document Frequency feature weighting.")
op.add_option("--use-hashing",action="store_true",default=False,help="using a hashing feature vectorizer")
op.add_option("--n-features",type=int,default=10000,help="Maximum number of features(dimensions to extract from text.)")
op.add_option("--verbose",action="store_true",dest="verbose",default=False,help="print report inside k-means")
op.print_help()
(opts,args) = op.parse_args()
if len(args) > 0:
    op.error(u"没有参数,请输入参数")
    sys.exit(1)

'''使用稀疏向量器选择特征向量训练数据'''
if opts.use_hashing:
    if opts.use_idf:
        hasher = HashingVectorizer(n_features=opts.n_features,stop_words='english', non_negative=True,
                                   norm=None, binary=False)
        '''使用pipeline创建一个管道'''
        vectorizer = make_pipeline(hasher, TfidfTransformer())
    else:
        vectorizer = HashingVectorizer(n_features=opts.n_features,
                                       non_negative=False, norm='l2',
                                       binary=False)

else:
    vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,min_df=2,use_idf=opts.use_idf)
X = vectorizer.fit_transform(X_train)



'''构建自定义减少特征维度的函数'''
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
if opts.n_components:
    reduce_dim = TruncatedSVD(opts.n_components)
    normalizer = Normalizer(copy=False)
    '''构建管道'''
    lsa = make_pipeline(reduce_dim,normalizer)

    X = lsa.fit_transform(X)

'''构建聚类函数'''
from sklearn.cluster import MiniBatchKMeans,KMeans
if opts.minibatch:
    km = MiniBatchKMeans(n_clusters=true_k,init='k-means++',n_init=1,init_size=1000,batch_size=1000,
                         verbose=opts.verbose)
else:
    km = KMeans(n_clusters=true_k,init='k-means++',max_iter=100,n_init=1,verbose=opts.verbose)
print ("Clustering sparse data with %s" % km)
km.fit(X)

'''输出元组数据'''
from sklearn import metrics
'''一致性'''
print ("一致性Homogeneity: %f" % metrics.homogeneity_score(y_train,km.labels_))
'''完整性'''
print("完整性Completeness: %f" % metrics.completeness_score(y_train, km.labels_))
'''v = 2 * (homogeneity * completeness) / (homogeneity + completeness)'''
print("V-measure: %f" % metrics.v_measure_score(y_train, km.labels_))
'''聚类指标RAND-INDEX'''
print("聚类评价指标Adjusted Rand-Index: %f" % metrics.adjusted_rand_score(y_train, km.labels_))
'''轮廓系数'''
print("轮廓系数Silhouette Coefficient: %f" % metrics.silhouette_score(X, km.labels_, sample_size=1000))

