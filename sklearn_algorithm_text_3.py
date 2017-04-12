#! -*-encoding=utf-8-*-
import numpy as np
n_topics = 10
n_topics_words = 10
n_features = 1000
X_train = [ ]
Y_train =[ ]
X_test = [ ]
Y_test =[ ]

'''构建函数的参数群'''
from optparse import OptionParser
import sys
op = OptionParser()
op.add_option("--report",action="store_true",dest="print_report",help=u"输出一个具体的分类器的类型")
op.add_option("--chi2_select",action="store",type="int",dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",action="store_true",dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class for every classifier.")
op.add_option("--all_categories",action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",action="store_true",help="Use a hashing vectorizer.")
op.add_option("--n_features",action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",action="store_true",
              help="Remove newsgroup information that is easily overfit: ""headers, signatures, and quoting.")
(opts,args) = op.parse_args()
if len(args) > 0:
    op.error(u"没有输入参数")
    sys.exit(1)
op.print_help()

'''读入数据'''
for line in open('F:\\WeiWeiHe\\data.txt'):
    X_train.append(line)
print X_train

for line in open('F:\\WeiWeiHe\\data_y.txt'):
    Y_train.append(line)
print Y_train

for line in open("F:\\WeiWeiHe\\datatest.txt"):
    X_test.append(line)
print X_test

for line in open("F:\\WeiWeiHe\\data_ytest.txt"):
    Y_test.append(line)
print Y_test

X_test = X_train
Y_test = Y_train
from sklearn.feature_extraction.text import HashingVectorizer,TfidfVectorizer
if opts.use_hashing:
    vectorizer = HashingVectorizer(non_negative=True,n_features=opts.n_features)
    X_train = vectorizer.transform(X_train)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True,max_df=0.5)
    X_train = vectorizer.fit_transform(X_train)
    X_test = vectorizer.transform(X_test)
print X_train.shape

from sklearn.feature_selection import SelectKBest,chi2
from sklearn import metrics
from sklearn.utils.extmath import density
'''if opts.select_chi2:
    ch2 = SelectKBest(chi2,k=opts.select_chi2)
    ch2.fit(X_train,Y_train)
    X_train = ch2.transform(X_train)
    X_test = ch2.transform(X_test)
'''

'''构造分类器'''
def benchmark(clf):
    print "TRAINING……"
    print (clf)
    clf.fit(X_train,Y_train)
    pred = clf.predict(X_test)
    score = metrics.accuracy_score(Y_test,pred)
    print ("accurancy: %f" % score)
    if hasattr(clf,'coef_'):
        print ("density: %f" % density(clf.coef_))
    if opts.print_report:
        print (u"分类报告：")
        print metrics.classification_report(Y_test,pred)
    if opts.print_cm:
        print (u"混沌矩阵：")
        print metrics.confusion_matrix(Y_test,pred)
        print (u"AUC：")
        print metrics.auc(Y_test, pred)
        print (u"召回率：")
        print metrics.recall_score(Y_test,pred)
        print (u"准确率：")
        print metrics.precision_score(Y_test, pred)
    clf_descr = str(clf).split('(')[0]
    return clf_descr,score

result =[]
from sklearn.linear_model import RidgeClassifier,PassiveAggressiveClassifier,Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
for clf, name in (
        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
        (Perceptron(n_iter=50), "Perceptron"),
        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
        (KNeighborsClassifier(n_neighbors=10), "kNN"),
        (RandomForestClassifier(n_estimators=100), "Random forest")):
    print('=' * 80)
    print(name)
    result.append(benchmark(clf))

from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import NearestCentroid
for penalty in ["l2", "l1"]:
    print('=' * 80)
    print("%s penalty" % penalty.upper())
    result.append(benchmark(LinearSVC(loss='l2', penalty=penalty,dual=False, tol=1e-3)))
    result.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,penalty=penalty)))

print('=' * 80)
print("Elastic-Net penalty")
result.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,penalty="elasticnet")))

print('=' * 80)
print("NearestCentroid (aka Rocchio classifier)")
result.append(benchmark(NearestCentroid()))

from sklearn.naive_bayes import MultinomialNB,BernoulliNB
# Train sparse Naive Bayes classifiers
print('=' * 80)
print("Naive Bayes")
result.append(benchmark(MultinomialNB(alpha=.01)))
result.append(benchmark(BernoulliNB(alpha=.01)))

from sklearn.pipeline import Pipeline
result.append(benchmark(Pipeline([
    ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
    ('classification', LinearSVC())
])))