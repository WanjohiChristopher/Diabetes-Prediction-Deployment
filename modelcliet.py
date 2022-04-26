import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report,accuracy_score


diabetes_df=pd.read_csv("diabetes.csv")

#predictor variables
X=diabetes_df.drop(columns=['Outcome'],axis=1)
#scaling our data
sc=StandardScaler()
X=sc.fit_transform(X)
#target variable
y=diabetes_df.Outcome

#splitting data
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=0,test_size=0.2)

# creating a balanced dataset
from imblearn.over_sampling import SMOTE
smt=SMOTE()
X_train,y_train=smt.fit_sample(X_train,y_train)

# we check the amount of records in each category
#np.bincount(y_train)

#logistic regression model

logi=LogisticRegression()
logi.fit(X_train,y_train)
#predictions
prediction=logi.predict(X_test)
#checking accuracy
print(accuracy_score(prediction,y_test))


pickle.dump(logi,open('diabetes.pkl','wb'))
