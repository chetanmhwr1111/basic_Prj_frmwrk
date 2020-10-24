import pandas as pd
df=pd.read_csv('Graduate Admissions.csv')
print(df.head())
print(df.columns)
df.rename(columns={'Serial No.':'S. No.','GRE Score':'GRE', 'TOEFL Score':'TOEFL','University Rating':'UR','Research':'Res','Chance of Admit ':'COA'},inplace=True)
print(df.columns)
print(df.head())
df.set_index('S. No.',inplace=True)
for i in df.index:
    if df.loc[i,'COA']>= .7:
        df.loc[i,'Admission Result']=1
    else:
        df.loc[i,'Admission Result']=0
print(df.head())
print(df.count())        
        

features=['GRE', 'TOEFL', 'UR', 'SOP', 'LOR ', 'CGPA', 'Res']
X=df[features]
Y=df['Admission Result']

from sklearn.linear_model import LogisticRegression 
lorm = LogisticRegression()
lorm.fit(X,Y)
R_sq=lorm.score(X,Y)
print('\n Coefficient of determination by Logistic Regression Model is:', "{0:.3f}".format(R_sq))

df['Predicting Admission Result']=lorm.predict(X)
print('\n', df.loc[:6,['Admission Result','Predicting Admission Result']] )
print('\n Logistic Regression Model Training Accuracy by evaluating on complete dataset:',mean(double(df['Predicting Admission Result'])==df['Admission Result'])*100)



from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)

from sklearn.linear_model import LogisticRegression 
lorm = LogisticRegression()
lorm.fit(X_train,Y_train)
R_sq=lorm.score(X_train,Y_train)
print('\n Coefficient of determination by Logistic Regression Model is:', "{0:.3f}".format(R_sq))

predict=pd.DataFrame()
predict['Predicting Admission Result']=lorm.predict(X_test)
print('\n',predict[0:10])
print('\n',Y_test[0:10])
print('\n Logistic Regression Model Test Accuracy by evaluating on test dataset:',mean(double(predict['Predicting Admission Result'])==Y_test)*100)



from sklearn.naive_bayes import GaussianNB
nbm=GaussianNB()
nbm.fit(X_train,Y_train)
R_sq=nbm.score(X_train,Y_train)
print('\n Coefficient of determination by Guassian Naive Bayes Model is:', "{0:.3f}".format(R_sq))

predict=pd.DataFrame()
predict['Predicting Admission Result']=lorm.predict(X_test)
print('\n',predict[0:10])
print('\n',Y_test[0:10])
print('\n Naive Bayes Model Test Accuracy by evaluating on test dataset:',mean(double(predict['Predicting Admission Result'])==Y_test)*100)



#POLY ORDER -2
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import PolynomialFeatures 
SS=StandardScaler()
X_train_scaled=SS.fit_transform(X_train)

poly=PolynomialFeatures(degree=2)
X_train_poly=poly.fit_transform(X_train_scaled)

SS=StandardScaler()
X_test_scaled=SS.fit_transform(X_test)

poly=PolynomialFeatures(degree=2)
X_test_poly=poly.fit_transform(X_test_scaled)

from sklearn.linear_model import LogisticRegression 
lorm = LogisticRegression()
lorm.fit(X_train_poly,Y_train)
R_sq=lorm.score(X_train_poly,Y_train)
print('\n Coefficient of determination by Logistic Regression Model is:', "{0:.3f}".format(R_sq))

predict=pd.DataFrame()
predict['Predicting Admission Result']=lorm.predict(X_test_poly)
print('\n Logistic Regression Model Test Accuracy by evaluating on test dataset:',mean(double(predict['Predicting Admission Result'])==Y_test)*100)


#POLY ORDER -3 (BEST)
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import PolynomialFeatures 
SS=StandardScaler()
X_train_scaled=SS.fit_transform(X_train)

poly=PolynomialFeatures(degree=3)
X_train_poly=poly.fit_transform(X_train_scaled)

SS=StandardScaler()
X_test_scaled=SS.fit_transform(X_test)

poly=PolynomialFeatures(degree=3)
X_test_poly=poly.fit_transform(X_test_scaled)

from sklearn.linear_model import LogisticRegression 
lorm = LogisticRegression()
lorm.fit(X_train_poly,Y_train)
R_sq=lorm.score(X_train_poly,Y_train)
print('\n Coefficient of determination by Logistic Regression Model is:', "{0:.3f}".format(R_sq))

predict=pd.DataFrame()
predict['Predicting Admission Result']=lorm.predict(X_test_poly)
print('\n Logistic Regression Model Test Accuracy by evaluating on test dataset:',mean(double(predict['Predicting Admission Result'])==Y_test)*100)

#POLY ORDER -4
from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import PolynomialFeatures 
SS=StandardScaler()
X_train_scaled=SS.fit_transform(X_train)

poly=PolynomialFeatures(degree=4)
X_train_poly=poly.fit_transform(X_train_scaled)

SS=StandardScaler()
X_test_scaled=SS.fit_transform(X_test)

poly=PolynomialFeatures(degree=4)
X_test_poly=poly.fit_transform(X_test_scaled)

from sklearn.linear_model import LogisticRegression 
lorm = LogisticRegression()
lorm.fit(X_train_poly,Y_train)
R_sq=lorm.score(X_train_poly,Y_train)
print('\n Coefficient of determination by Logistic Regression Model is:', "{0:.3f}".format(R_sq))

predict=pd.DataFrame()
predict['Predicting Admission Result']=lorm.predict(X_test_poly)
print('\n Logistic Regression Model Test Accuracy by evaluating on test dataset:',mean(double(predict['Predicting Admission Result'])==Y_test)*100)


