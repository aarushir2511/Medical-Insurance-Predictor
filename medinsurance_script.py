# -*- coding: utf-8 -*-
"""MedInsurance_Script.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1l6p6yTBY9JIMXcvwOOiNpquaPQeQByTE
"""

import pandas as  pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

df=pd.read_csv('medical_insurance.csv')

df.head()

df.shape

df.dtypes

df.info()

df['charges']=df['charges'].round(2)

df['charges'].head()

df['sex'].unique()

df['charges'].isnull().sum()

df.isnull().sum()

df['age'].mean()

df['charges'].mean().round(2)

df['region'].max()

df['bmi'].mean()

df['smoker'].value_counts()

df['region'].value_counts().plot(kind='bar', color ='midnightblue')
plt.xlabel('Region')
plt.ylabel('Count')
plt.title('Distribution of Regions')
plt.show()

df['sex'].value_counts().plot(kind='pie',autopct='%1.1f%%',colors=('blue','deepskyblue'))
plt.legend()
plt.show()

df['smoker'].value_counts().plot(kind='pie',autopct='%1.1f%%',colors=('deepskyblue','blue'))
plt.legend()
plt.show()

df['children'].unique()

df[df['children'] == 5]['children'].count()

df['children'].value_counts().plot(kind='bar',color='midnightblue')
plt.xlabel('Children')
plt.ylabel('Count')
plt.title('Distribution of Children')
plt.show()

fig, ax1 = plt.subplots(figsize=(10,8))
ax1.scatter(df['age'],df['charges'],color='b',marker='o')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Age vs Charges')
plt.show()

fig,ax1=plt.subplots(figsize=(10,8))
lb=df['charges'].quantile(0.10)
ub=df['charges'].quantile(0.90)
df['charges']=df['charges'].clip(lb,ub)
ax1.scatter(df['region'],df['charges'],color='b',marker='o')
plt.xlabel('Region')
plt.ylabel('Charges')
plt.title('Region vs Charges')
plt.show()

fig,ax1=plt.subplots(figsize=(10,8))
ax1.scatter(df['smoker'],df['charges'],color='b',marker='o')
plt.xlabel('smoker')
plt.ylabel('Charges')
plt.title('smoker vs Charges')
plt.show()

fig,ax1=plt.subplots(figsize=(10,6))
ax1.scatter(df['bmi'],df['charges'],color='b',marker='o')
plt.xlabel('bmi')
plt.ylabel('Charges')
plt.title('bmi vs Charges')
plt.show()

fig,ax1=plt.subplots(figsize=(10,6))
ax1.scatter(df['sex'],df['charges'],color='b',marker='o')
plt.xlabel('sex')
plt.ylabel('Charges')
plt.title('sex vs Charges')
plt.show()

fig,ax1=plt.subplots(figsize=(10,6))
ax1.scatter(df['children'],df['charges'],color='b',marker='o')
plt.xlabel('children')
plt.ylabel('Charges')
plt.title('children vs Charges')
plt.show()

df_corr=df.copy()
label_enc = LabelEncoder()
df_corr['sex']=label_enc.fit_transform(df_corr['sex'])
df_corr['smoker']=label_enc.fit_transform(df_corr['smoker'])
df_corr['region']=label_enc.fit_transform(df_corr['region'])
df_corr.head(10)

df_corr2=df_corr.corr()
df_corr2

sns.heatmap(df_corr2,annot=True,cmap='magma')
plt.show()

X=df_corr.drop('charges',axis=1)
Y=df_corr['charges']

X.head()

Y.head()

X_train,X_test, Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_train

X_test_scaled = scaler.transform(X_test)
X_test

"""**LR**"""

lr = LinearRegression()
lr.fit(X_train_scaled,Y_train)

print("Shape of X_train_scaled:", X_train_scaled.shape)
print("Shape of Y_train:", Y_train.shape)

Y_pred = lr.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred))
rmse

r_2= r2_score(Y_test, Y_pred)
r_2

plt.figure(figsize=(10, 6))
plt.scatter(Y_test, Y_pred, color='blue', alpha=0.8, label='Data Points')
plt.plot([min(Y_test), max(Y_test)], [min(Y_test), max(Y_test)], color='red',linewidth=2, linestyle='--')

plt.xlabel("Actual Charges")
plt.ylabel("Predicted Charges")
plt.title("Actual vs Predicted Charges - Linear Regression")
plt.legend()
plt.show()

"""**DT**"""

dtr = DecisionTreeRegressor(random_state=42)

dtr.fit(X_train_scaled,Y_train)

Y_pred_dtr = dtr.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred_dtr))
rmse

r2 = r2_score(Y_test, Y_pred_dtr)
r2

plt.scatter(Y_test, Y_pred_dtr, color='blue', alpha=0.8, label='Data Points')
plt.plot([min(Y_test),max(Y_test)] , [min(Y_test),max(Y_test)], color='red',linewidth=2, linestyle='--')
plt.xlabel("Actual charges")
plt.ylabel("Predicted charges")
plt.title("Actual vs Predicted Charges - Decision Tree Regression")
plt.legend()
plt.show()

"""**RFG**"""

rfr = RandomForestRegressor(n_estimators=150, random_state=42)

rfr.fit(X_train_scaled,Y_train)

Y_pred_rfr=rfr.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred_rfr))
rmse

r2 = r2_score(Y_test, Y_pred_rfr)
r2

plt.scatter(Y_test, Y_pred_rfr, alpha = 0.8, color = 'blue', label='Data Points')
plt.plot([min(Y_test), max(Y_test)] , [min(Y_test), max(Y_test)], color='red',linewidth=2, linestyle='--')
plt.xlabel("Actual charges")
plt.ylabel("Predicted charges")
plt.title("Actual vs Predicted Charges - Random Forest Regression")
plt.legend()
plt.show()

"""**KNN**"""

knn = KNeighborsRegressor(n_neighbors=3)

knn.fit(X_train_scaled,Y_train)

Y_pred_knn=knn.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred_knn))
rmse

r2 = r2_score(Y_test, Y_pred_knn)
r2

plt.scatter(Y_test, Y_pred_knn, alpha = 0.8, color = 'blue', label='Data Points')
plt.plot([min(Y_test), max(Y_test)] , [min(Y_test), max(Y_test)], color='red',linewidth=2,linestyle='--')
plt.xlabel("Actual charges")
plt.ylabel("Predicted charges")
plt.title("Actual vs Predicted Charges - KNN Regression")
plt.legend()
plt.show()

"""**SVR**"""

from sklearn.pipeline import make_pipeline

svr = make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, epsilon=0.1))

svr.fit(X_train_scaled,Y_train)

Y_pred_svr=svr.predict(X_test_scaled)

rmse = np.sqrt(mean_squared_error(Y_test, Y_pred_svr))
rmse

r2 = r2_score(Y_test, Y_pred_svr)
r2

plt.scatter(Y_test, Y_pred_svr, alpha = 0.8, color = 'blue', label='Data Points')
plt.plot([min(Y_test), max(Y_test)] , [min(Y_test), max(Y_test)], color='red',linewidth=2, linestyle='--')
plt.xlabel("Actual charges")
plt.ylabel("Predicted charges")
plt.title("Actual vs Predicted Charges - SVR Regression")
plt.legend()
plt.show()

"""**From all the models above we see that the best is RanomForestRegressor**"""

def prediction_rfr(new_input):
  prediction = rfr.predict(new_input)
  prediction.astype(int)
  return prediction

n = [60, 0, 25.8, 0, 0, 1]
#age	sex	bmi	children	smoker	region	charges

print(prediction_rfr([n])[0])

'''	31	0	25.740	0	0	2	3756.62
   	46	0	33.440	1	0	2	8240.59'''

X_train, X_remaining, y_train, y_remaining = train_test_split(X, Y, test_size=0.4, random_state=42)
X_cv, X_test, y_cv, y_test = train_test_split(X_remaining, y_remaining, test_size=0.5, random_state=42)

rfr2 = RandomForestRegressor(n_estimators=150, random_state=42)

rfr2.fit(X_train, y_train)

y_cv_pred = rfr2.predict(X_cv)

y_test_pred = rfr2.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_cv, y_cv_pred))
rmse

def prediction_rfr_pt2(new_input):
  prediction = rfr2.predict(new_input)
  prediction = prediction.round(2)
  return print(prediction)

n = [19, 0, 28, 0, 1, 3]
#age	sex	bmi	children	smoker	region	charges

prediction_rfr_pt2([n])

import pickle as pkl

filename = 'med_model_rfr.sav'
with open(filename, 'wb') as file:
  pkl.dump(rfr2, file)

import sklearn
print(sklearn.__version__)