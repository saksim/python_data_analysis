#! -*-encoding=utf-8-*-
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
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
'''算法'''
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(enc.transform(X_train),Y_train)

'''查看准确率'''
Y_score = logreg.score(enc.transform(X_train),Y_train)
Y_train_logreg = logreg.predict_proba(enc.transform(X_train))[:,1]
Y_train_label = logreg.predict(enc.transform(X_train))
'''以上训练完毕，现在开始使用测试数据，对测试数据'''
'''概率估计'''
y_predict_logreg = logreg.predict_proba(enc.transform(X_test))[:,1]
'''对已给特征向量进行分类分类进行预测'''
y_predict_label = logreg.predict(enc.transform(X_test))
Y_test_label = pd.DataFrame(y_predict_label,columns=['survived'])
print Y_test_label

'''输出 还没有解决列的替换问题'''
'''result =df22.drop('survived',1)
result = pd.merge(result,y_predict_label,on=index)
print result
'''

'''ROC'''
from sklearn.metrics import roc_curve
fpr,tpr,_= roc_curve(Y_train,Y_train_label)
print fpr
print tpr

'''画图'''
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr, label=u'基于泰坦尼克号数据使用逻辑回归分析')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

