#! -*-encoding=utf-8-*-
import matplotlib.pyplot as plt
'''from matplotlib import style
style.use('ggplot')'''
import numpy as np
from sklearn import preprocessing
import pandas as pd

'''前一批数据'''
df11 = pd.read_excel('F:\\WeiWeiHe\\titanic.xls')
'''df.convert_objects(convert_numeric=True)'''''
'''df.fillna(0, inplace=True)'''

'''这一批数据'''
df22 =pd.read_excel('F:\\WeiWeiHe\\titanic_new.xls')

'''将数据汇总作为总数据'''
df_all = df11.append(df22,ignore_index=True)

'''2)消除缺失值'''
df_all.fillna('none',inplace=True)
print df_all
'''
from sklearn.preprocessing import Imputer
imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
df = imp.fit_transform(df)
df = pd.DataFrame(df,columns=df_all.columns)
print df
'''

'''先转换为纯数字型'''
from sklearn.preprocessing.label import LabelEncoder
class_label_encoder = LabelEncoder()
df = df_all.values
df1 = df11.values
df2 = df22.values
for i in range(14):
    df[:,i] = class_label_encoder.fit_transform(df[:,i])
    print df[:,i]
    df1[:,i] = class_label_encoder.fit_transform(df1[:,i])
    df2[:,i] = class_label_encoder.fit_transform(df2[:,i])
df = pd.DataFrame(df,columns=df_all.columns)

'''划分数据集'''
'''
NUMBER = 1309
X_train = df[:NUMBER].drop(['survived'],1)
X_test = df[NUMBER:].drop(['survived'],1)
survivor = df['survived']
Y_train = survivor[:NUMBER]
Y_test = survivor[NUMBER:]
print Y_test
'''
NUMBER = 1309
X = df.drop('survived',1)
X_train = df[:NUMBER].drop(['survived'],1)
X_test = df[NUMBER:].drop(['survived'],1)
survivor = df['survived']
Y_train = np.array(survivor[:NUMBER]).astype(float)

'''print Y_train.dtype'''

'''转换为归一化矩阵数据'''
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
enc.fit(X)
print enc.transform(X_train).toarray().shape
'''************************以上都一样************************'''

'''算法'''
'''
1)逻辑回归做分类
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(enc.transform(X_train),Y_train)

Y_score = logreg.score(enc.transform(X_train),Y_train)
Y_train_logreg = logreg.predict_proba(enc.transform(X_train))
Y_train_label = logreg.predict(enc.transform(X_train))

from sklearn.metrics import roc_curve,auc
fpr,tpr,thresholds = roc_curve(Y_train,Y_train_logreg[:,1])
roc_auc = auc(fpr,tpr)

from sklearn.metrics import roc_auc_score
auc_result = roc_auc_score(Y_train,Y_train_logreg[:,1])
print auc_result
print roc_auc

plt.plot(fpr,tpr,label = 'ROC fold')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(u'基于')
plt.legend(loc="lower right")
plt.show()

train_index = logreg.coef_
print train_index
'''
''' 支持向量机C SVC
from sklearn.svm import SVC
svc = SVC(probability = True)
svc.fit(enc.transform(X_train),Y_train)

Y_train_logreg = svc.predict_proba(enc.transform(X_train))
Y_train_label = svc.predict(enc.transform(X_train))

from sklearn.metrics import roc_curve,auc
fpr,tpr,thresholds = roc_curve(Y_train,Y_train_logreg[:,1])
roc_auc = auc(fpr,tpr)

from sklearn.metrics import roc_auc_score
auc_result = roc_auc_score(Y_train,Y_train_logreg[:,1])
print auc_result
print roc_auc

plt.plot(fpr,tpr,label = 'ROC fold')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(u'基于泰坦尼克号存活名单数据集，使用SVC进行数据拟合的ROC曲线')
plt.legend(loc="lower right")
plt.show()
'''
'''
神经网络分类器MLP
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(verbose=0, random_state=0,max_iter=400)
mlp.fit(enc.transform(X_train),Y_train)

Y_train_pro = mlp.predict_proba(enc.transform(X_train))
Y_test_pro = mlp.predict_proba(enc.transform(X_test))
Y_test_label = mlp.predict(enc.transform(X_test))

from sklearn.metrics import roc_curve,auc
fpr,tpr,_ = roc_curve(Y_train,Y_train_pro[:,1])
NN_auc = auc(fpr,tpr)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label=u'基于泰坦尼克号数据,使用神经网络MLP分类器进行分类')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
'''


