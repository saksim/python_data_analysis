#! -*-encoding=utf-8-*-
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

df = pd.read_excel('F:\\WeiWeiHe\\titanic.xls')
df.convert_objects(convert_numeric=True)
'''df.fillna(0, inplace=True)'''
print df

unique_name_lable = np.unique(df['name'])
unique_sex_label = np.unique(df['sex'])
unique_ticket_label = np.unique(df['ticket'])
unique_cabin_label = np.unique(df['cabin'])
unique_embarked_label = np.unique(df['embarked'])
unique_boat_label = np.unique(df['boat'])
unique_address_label = np.unique(df['home.dest'])

name_mapping = {label:idx for idx,label in enumerate(unique_name_lable)}
sex_mapping = {label:idx for idx,label in enumerate(unique_sex_label)}
ticket_mapping = {label:idx for idx,label in enumerate(unique_ticket_label)}
cabin_mapping = {label:idx for idx,label in enumerate(unique_cabin_label)}
embarked_mapping = {label:idx for idx,label in enumerate(unique_embarked_label)}
boat_mapping = {label:idx for idx,label in enumerate(unique_boat_label)}
address_mapping = {label:idx for idx,label in enumerate(unique_address_label)}

print name_mapping
print sex_mapping
print ticket_mapping
print cabin_mapping
print embarked_mapping
print boat_mapping
print address_mapping

df['name'] = df['name'].map(name_mapping)
df['sex'] = df['sex'].map(sex_mapping)
df['ticket'] = df['ticket'].map(ticket_mapping)
df['cabin'] = df['cabin'].map(cabin_mapping)
df['embarked'] = df['embarked'].map(embarked_mapping)
df['boat'] = df['boat'].map(boat_mapping)
df['home.dest'] = df['home.dest'].map(address_mapping)
df.fillna(0, inplace=True)

print df

'''

samples = df
print samples
samples = [dict(enumerate(sample)) for sample in samples]
print samples

from sklearn.feature_extraction import DictVectorizer
vect = DictVectorizer(sparse=False)
X = vect.fit_transform(samples)
print X



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
print X.shape
y = np.array(df['survived'])
print y.shape
clf = KMeans(n_clusters=2,random_state=None).fit(X)
print clf.labels_

correct = 0
for i in range(len(X)):
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))

    prediction = clf.predict(predict_me)
    print prediction
    if prediction == y[i]:
        correct += 1

percent1 = float(correct)
percent2 = float(len(X))
print percent1
print percent2
print '%f'%(percent1/percent2)
print (float(correct)/float(len(X)))

from sklearn.externals import joblib
joblib.dump(clf, 'F:\\WeiWeiHe\\KMeans.pkl')
newmodel = joblib.load('F:\\WeiWeiHe\\KMeans.pkl')
df2 = pd.DataFrame([3,"WeiWeiHE",'male',28,0,0,41813,176.2917,'A15','C',8,None,'China'])
handle_non_numerical_data(df2)
df2 = np.array(df2).astype(float)
df2 = df2.reshape(-1,len(df2))
print newmodel.predict(df2)
'''