#! -*-encoding=utf-8-*-
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn import preprocessing
import pandas as pd

df = pd.read_excel('F:\\WeiWeiHe\\titanic.xls')
'''df.convert_objects(convert_numeric=True)'''''
'''df.fillna(0, inplace=True)'''

from sklearn.preprocessing.label import LabelEncoder
class_label_encoder = LabelEncoder()
print df.head()
df = df.values
df[:,2] = class_label_encoder.fit_transform(df[:,2])
df[:,3] = class_label_encoder.fit_transform(df[:,3])
df[:,7] = class_label_encoder.fit_transform(df[:,7])
df[:,9] = class_label_encoder.fit_transform(df[:,9])
df[:,10] = class_label_encoder.fit_transform(df[:,10])
df[:,11] = class_label_encoder.fit_transform(df[:,11])
df[:,13] = class_label_encoder.fit_transform(df[:,13])

from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN',strategy='mean',axis=1)
df = imp.fit_transform(df)
df = pd.DataFrame(df,columns=['pclass','survived','name','sex','age','sibsp','parch','ticket','fare','cabin','embarked','boat','body','home.dest'])
print df

'''
使用平均数填充缺失值
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN',strategy='mean',axis=1)
df1 = imp.fit_transform(df)
df1 = pd.DataFrame(df1,columns=['pclass','survived','name','sex','age','sibsp','parch','ticket','fare','cabin','embarked','boat','body','home.dest'])
print df1

使用0填充缺失值
df = pd.DataFrame(df,columns=['pclass','survived','name','sex','age','sibsp','parch',
'ticket','fare','cabin','embarked','boat','body','home.dest'])
df.fillna(0, inplace=True)
print df
'''
'''df1 = pd.DataFrame(df.reshape(len(df),-1))'''

'''使用SVM算法'''
'''
X = np.array(df.drop('survived',1))
X = preprocessing.scale(X)
print X.shape
y = np.array(df['survived'])
print y.shape
from sklearn import svm
clf = svm.SVC()
clf.fit(X,y)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))

    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct += 1

print (float(correct)/float(len(X)))
'''
'''
X = np.array(df.drop('survived',1))
X = preprocessing.scale(X)
print X.shape
y = np.array(df['survived'])
print y.shape
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X,y)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))

    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct += 1

print (float(correct)/float(len(X)))
'''

from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn import decomposition
from sklearn.pipeline import Pipeline

logistic = LogisticRegression()
pca = decomposition.PCA()
pipe = Pipeline(steps=[('pca',pca),('logistic',logistic)])

X = np.array(df.drop('survived',1))
X = preprocessing.scale(X)
print X.shape
y = np.array(df['survived'])
print y.shape
clf = pca.fit_transform(X,y)
plt.figure(1,figsize=(5,5))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_')
n_components = [1, 2,3,4,5,6,7,8,9,10,11,12,13]
Cs = np.logspace(-4, 4, 3)
estimator = GridSearchCV(pipe,
dict(pca__n_components=n_components,
logistic__C=Cs))
estimator.fit(X, y)
plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
linestyle=':', label=u'最好的选择的特征成分个数')
plt.legend(prop=dict(size=12))
plt.show()