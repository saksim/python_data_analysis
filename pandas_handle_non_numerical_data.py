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