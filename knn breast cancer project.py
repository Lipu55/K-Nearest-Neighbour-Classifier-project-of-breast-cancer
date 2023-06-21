# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 21:33:08 2023

@author: MRUTYUNJAY
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
df=pd.read_csv(r"D:\Datascience Classes\30,31th march\projects\KNN\brest cancer.txt")
df.shape
df.head()
col_names=['Id','Clump_thickness','Uniformity_Cell_size','Uniformity_Cell_Shape','Marginal_Adhesion','Single_Epethelial_Cell_Size','Bare_Nuclei','Bland_Chromatin','Normal_Nucleoli','Mitoses','Class']
df.columns=col_names
df.columns
df.head()
df.drop('Id',axis=1,inplace=True)
df.info()
for var in df.columns:
    print(df[var].value_counts())
df['Bare_Nuclei']=pd.to_numeric(df['Bare_Nuclei'],errors='coerce')    
df.dtypes
df.isnull().sum()
df.isna().sum()
df['Bare_Nuclei'].value_counts()
df['Bare_Nuclei'].unique()
df['Bare_Nuclei'].isna().sum()
df['Class'].value_counts()
df['Class'].value_counts()/np.float(len(df))
print(round(df.describe(),2))
plt.rcParams['figure.figsize']=(30,25)
df.plot(kind='hist',bins=10,subplots=True,layout=(5,2),sharex=False,sharey=False)
plt.show()
correlation=df.corr()
correlation['Class'].sort_values(ascending=False)
plt.figure(figsize=(10,8))
plt.title('Correlation of Attributes with Class variable')
a=sns.heatmap(correlation,square=True,annot=True,fmt='.2f',linecolor='white')
a.set_xticklabels(a.get_xticklabels(),rotation=90)
a.set_yticklabels(a.get_yticklabels(),rotation=30)
plt.show()
X=df.drop(['Class'],axis=1)
y=df['Class']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
X_train.shape,X_test.shape
X_train.dtypes
X_train.isnull().sum()
X_test.isnull().sum()
for col in X_train.columns:
    if X_train[col].isnull().mean()>0:
        print(col,round(X_train[col].isnull().mean(),4))
for df1 in [X_train,X_test]:
    for col in X_train.columns:
        col_median=X_train[col].median()
        df1[col].fillna(col_median,inplace=True)
X_train.isnull().sum()        
X_test.isnull().sum()
X_train.head()
X_test.head()
cols=X_train.columns
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_train=pd.DataFrame(X_train,columns=[cols])
X_test=pd.DataFrame(X_test,columns=[cols])
X_train.head()
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
y_pred=knn.predict(X_test)
y_pred
knn.predict_proba(X_test)[:,0]
knn.predict_proba(X_test)[:,1]
from sklearn.metrics import accuracy_score
print('Model accuracy score: {0:0.4f}'. format(accuracy_score(y_test,y_pred)))
y_pred_train=knn.predict(X_train)
print('Training-set accuracy score: {0:0.4f}'. format(accuracy_score(y_train,y_pred_train)))
print('Training set score: {:.4f}'.format(knn.score(X_train,y_train)))
print('Test set score: {:,.4f}'.format(knn.score(X_test,y_test)))
y_test.value_counts()
null_accuracy=(85/(85+55))
print('Null accuracy score: {0:0.4f}'. format(null_accuracy))
knn_5=KNeighborsClassifier(n_neighbors=5)
knn_5.fit(X_train,y_train)
y_pred_5=knn_5.predict(X_test)
print('Model accuracy score with k=5 : {0:0.4f}'. format(accuracy_score(y_test,y_pred_5)))
knn_6=KNeighborsClassifier(n_neighbors=6)
knn_6.fit(X_train,y_train)
y_pred_6=knn_6.predict(X_test)
print('Model accuracy score with k=6 : {0:0.4f}'. format(accuracy_score(y_test,y_pred_6)))
knn_7=KNeighborsClassifier(n_neighbors=7)
knn_7.fit(X_train,y_train)
y_pred_7=knn_7.predict(X_test)
print('Model accuracy score with k=7 : {0:0.4f}'. format(accuracy_score(y_test,y_pred_7)))
knn_8=KNeighborsClassifier(n_neighbors=8)
knn_8.fit(X_train,y_train)
y_pred_8=knn_8.predict(X_test)
print('Model accuracy score with k=8 : {0:0.4f}'. format(accuracy_score(y_test,y_pred_8)))
knn_9=KNeighborsClassifier(n_neighbors=9)
knn_9.fit(X_train,y_train)
y_pred_9=knn_9.predict(X_test)
print('Model accuracy score with k=9 : {0:0.4f}'. format(accuracy_score(y_test,y_pred_9)))
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print('Confusion matrix\n\n',cm)
print('\nTrue Positives(TP) =',cm[0,0])
print('\nTrue Negatives(TN) =',cm[1,1])
print('\nFalse Positives(FP) =',cm[0,1])
print('\nFalse Negatives(FN) =',cm[1,0])
cm_7 = confusion_matrix(y_test, y_pred_7)

print('Confusion matrix\n\n', cm_7)

print('\nTrue Positives(TP) = ', cm_7[0,0])

print('\nTrue Negatives(TN) = ', cm_7[1,1])

print('\nFalse Positives(FP) = ', cm_7[0,1])

print('\nFalse Negatives(FN) = ', cm_7[1,0])
plt.figure(figsize=(6,4))
cm_matrix=pd.DataFrame(data=cm_7, columns=['Actual Positive:1','Actual Negative:0'],index=['Predict Positive:1','Predict Negative:0'])
sns.heatmap(cm_matrix,annot=True,fmt='d',cmap='YlGnBu')
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred_7))
TP = cm_7[0,0]
TN = cm_7[1,1]
FP = cm_7[0,1]
FN = cm_7[1,0]
classification_accuracy = (TP + TN) / float(TP + TN + FP + FN)
print('Classification accuracy : {0:0.4f}'.format(classification_accuracy))
classification_error = (FP + FN) / float(TP + TN + FP + FN)
print('Classification error : {0:0.4f}'.format(classification_error))
precision = TP / float(TP + FP)
print('Precision : {0:0.4f}'.format(precision))
recall = TP / float(TP + FN)
print('Recall or Sensitivity : {0:0.4f}'.format(recall))
true_positive_rate = TP / float(TP + FN)
print('True Positive Rate : {0:0.4f}'.format(true_positive_rate))
false_positive_rate = FP / float(FP + TN)
print('False Positive Rate : {0:0.4f}'.format(false_positive_rate))
specificity = TN / (TN + FP)
print('Specificity : {0:0.4f}'.format(specificity))
y_pred_prob = knn.predict_proba(X_test)[0:10]
y_pred_prob
y_pred_prob_df = pd.DataFrame(data=y_pred_prob, columns=['Prob of - benign cancer (2)', 'Prob of - malignant cancer (4)'])
y_pred_prob_df
knn.predict_proba(X_test)[0:10, 1]
y_pred_1 = knn.predict_proba(X_test)[:, 1]
y_pred_1
plt.figure(figsize=(6,4))
# adjust the font size 
plt.rcParams['font.size'] = 12
# plot histogram with 10 bins
plt.hist(y_pred_1, bins = 10)
# set the title of predicted probabilities
plt.title('Histogram of predicted probabilities of malignant cancer')
# set the x-axis limit
plt.xlim(0,1)
# set the title
plt.xlabel('Predicted probabilities of malignant cancer')
plt.ylabel('Frequency')
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_1, pos_label=4)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, linewidth=2)
plt.plot([0,1], [0,1], 'k--' )
plt.rcParams['font.size'] = 12
plt.title('ROC curve for Breast Cancer kNN classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.show()
from sklearn.metrics import roc_auc_score
ROC_AUC = roc_auc_score(y_test, y_pred_1)
print('ROC AUC : {:.4f}'.format(ROC_AUC))
from sklearn.model_selection import cross_val_score
Cross_validated_ROC_AUC = cross_val_score(knn_7, X_train, y_train, cv=5, scoring='roc_auc').mean()
print('Cross validated ROC AUC : {:.4f}'.format(Cross_validated_ROC_AUC))
from sklearn.model_selection import cross_val_score
scores = cross_val_score(knn_7, X_train, y_train, cv = 10, scoring='accuracy')
print('Cross-validation scores:{}'.format(scores))

print('Average cross-validation score: {:.4f}'.format(scores.mean()))
