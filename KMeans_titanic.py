#! -*-encoding=utf-8-*-
__author__ = 'heweiwei'

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

'''
Pclass Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)
survival Survival (0 = No; 1 = Yes)
name Name
sex Sex
age Age
sibsp Number of Siblings/Spouses Aboard
parch Number of Parents/Children Aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''
df = pd.read_csv('F:\\WeiWeiHe\\titanic.csv')
'''df = pd.read_excel('titanic.xls')'''
#print(df.head())
df.drop(['body','name'], 1, inplace=True)
df.convert_objects(convert_numeric=True)
'''
convert_objects是为对象列寻找更合适的类型,但是现在已经变为了to_numeric
当convert_numeric为真,
强制转换为 numbers (包括 strings),不能转换的转为 NaN
返回的对象和输入的对象同一类型

'''
df.fillna(0, inplace=True)
'''
fillna会寻找DataFrame中的NA/NaN值。
参数值：一般为0，是用于弥补NA/NaN的值
method:弥补方法（‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None）
'''
#print(df.head())

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            column_contents = df[column].values.tolist()
            unique_elements = set(column_contents)
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))

    return df

df = handle_non_numerical_data(df)

X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=10)
clf.fit(X)
print clf.fit(X)
correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    '''numpy中reshape，如果对一维度的参数为-1，则忽略这一维度，按照另外的维度参数计算。比如有12个数，为(3,4),如果reshape(-1,2),则会变成(6,2)的数组'''
    prediction = clf.predict(predict_me)
    if abs(prediction[0] - y[i]) <= 0.1:
        correct += 1

print prediction
print (correct/len(X))
print X[1]

'''
correct = 0
for i in range(len(X)):
    predict_me = np.array(X(i).astype(float))  这里写错了！X[i]才是正确格式，而不是X(i)
    predict_me = predict_me.reshape(-1,len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct +=1
print (correct/len(X))
'''

'''
dfcsv = df.to_csv('F:\\WeiWeiHe\\titanic_handled1.csv',index = False)
'''