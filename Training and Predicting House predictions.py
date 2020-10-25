import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt
df=pd.read_csv('kc_house_data_NaN.csv')
df.head()

column_1=df.columns[0]
column_2=df.columns[1]
df.drop([column_1,column_2],axis=1,inplace=True)

df=df.dropna()
df=df.reset_index()


#1
clf()
sns.boxplot(x=df['waterfront'],y=df['price'])
plt.show()
plt.title('House Price dependency on Waterfront')

import scipy.stats as ss
Pearson_coef,p_value=ss.pearsonr(df['waterfront'],df['price'])
print("Pearson correlation coefficient:","{0:.1f}".format(Pearson_coef))
print("p_value :",p_value) 
#2
clf()
sns.regplot(x=df['sqft_living'],y=df['price'])
plt.ylim(0,)
plt.title('House Price dependency on Flat area')

Pearson_coef,p_value=ss.pearsonr(df['sqft_living'],df['price'])
print("Pearson correlation coefficient:","{0:.1f}".format(Pearson_coef))
print("p_value :",p_value) 
#3
clf()
sns.regplot(x=df['sqft_lot'],y=df['price'])
xlim(0,200000)
plt.ylim(0,)
plt.title('House Price dependency on Plot area')

Pearson_coef,p_value=ss.pearsonr(df['sqft_lot'],df['price'])
print("Pearson correlation coefficient:","{0:.1f}".format(Pearson_coef))
print("p_value :",p_value) 
#4
clf()
sns.regplot(y=df['sqft_living'],x=df['sqft_lot'])
xlim(0,200000)
plt.ylim(0,)
xlabel('Plot area')
ylabel('Flat area')
plt.title('Relation of Flat area and Plot area')

Pearson_coef,p_value=ss.pearsonr(df['sqft_lot'],df['sqft_living'])
print("Pearson correlation coefficient:","{0:.1f}".format(Pearson_coef))
print("p_value :",p_value) 
#5
clf()
Pdf=df[['bedrooms','bathrooms','waterfront','view']]
sns.boxplot('bedrooms','bathrooms',data=Pdf,hue='waterfront')
plt.ylim(0,)
xlim(0,11)
plt.title('Relation between Bedrooms and Bathrooms for different waterfront')
#6
clf()
sns.regplot(x=df['bedrooms'],y=df['price'])
plt.ylim(0,)
xlim(0,11)
plt.title('House Price dependency on Bedrooms')

Pearson_coef,p_value=ss.pearsonr(df['bedrooms'],df['price'])
print("Pearson correlation coefficient:","{0:.1f}".format(Pearson_coef))
print("p_value :",p_value) 
#7
clf()
Pdf=df[['sqft_living','price','waterfront','view']]
g=sns.lmplot('sqft_living','price',data=Pdf,fit_reg=False,col='waterfront')
plt.ylim(0,)
plt.subplots_adjust(top=.9)
g.fig.suptitle('House Price dependency on Plot area for varied waterfront')
#8
clf()
Pdf=df[['sqft_living','price','waterfront','view']]
g=sns.FacetGrid(Pdf,col='view',row='waterfront',margin_titles=True)
g.map(plt.scatter,'sqft_living','price')
plt.subplots_adjust(top=.9)
g.fig.suptitle('House Price dependency on various factors')

df.groupby('view')['sqft_lot'].mean()


features =["floors","waterfront","lat","bedrooms","sqft_basement","view","bathrooms","sqft_living15","sqft_above","grade","sqft_living"]
X=df[features]
Y=df['price']

from sklearn.linear_model import LinearRegression 
lrm = LinearRegression()
lrm.fit(X,Y)
R_sq=lrm.score(X,Y)
print('coefficient of determination by Linear Regression Model is:', "{0:.3f}".format(R_sq))

from sklearn.model_selection import train_test_split
X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.3)

from sklearn.linear_model import LinearRegression 
lrm = LinearRegression()
lrm.fit(X_train,Y_train)
R_sq=lrm.score(X_train,Y_train)
print('coefficient of determination by Regression Model for Polynomial feature is:', "{0:.3f}".format(R_sq))





from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import PolynomialFeatures 
SS=StandardScaler()
X_train_scaled=SS.fit_transform(X_train)

SS=StandardScaler()
X_test_scaled=SS.fit_transform(X_test)

poly=PolynomialFeatures(degree=2)
X_train_poly=poly.fit_transform(X_train_scaled)

poly=PolynomialFeatures(degree=2)
X_test_poly=poly.fit_transform(X_test_scaled)


from sklearn.linear_model import LinearRegression 
lrm = LinearRegression()
lrm.fit(X_train_poly,Y_train)
R_sq=lrm.score(X_train_poly,Y_train)
print('coefficient of determination by Regression Model for Polynomial feature is:', "{0:.3f}".format(R_sq))

from sklearn.linear_model import Ridge 
modl =Ridge(alpha=.1)
modl.fit(X_train_poly,Y_train)
R_sq=modl.score(X_train_poly,Y_train)
print('coefficient of determination by Ridge Model for Polynomial feature is:', "{0:.3f}".format(R_sq))

from sklearn.linear_model import Ridge 
modl =Ridge(alpha=.01)
modl.fit(X_train_poly,Y_train)
R_sq=modl.score(X_train_poly,Y_train)
print('coefficient of determination by Ridge Model for Polynomial feature is:', "{0:.3f}".format(R_sq))





poly=PolynomialFeatures(degree=3)
X_train_poly=poly.fit_transform(X_train_scaled)

poly=PolynomialFeatures(degree=3)
X_test_poly=poly.fit_transform(X_test_scaled)

from sklearn.linear_model import Ridge 
modl =Ridge(alpha=.1)
modl.fit(X_train_poly,Y_train)
R_sq=modl.score(X_train_poly,Y_train)
print('coefficient of determination by Ridge Model for Polynomial feature is:', "{0:.3f}".format(R_sq))



from sklearn.preprocessing import StandardScaler  
from sklearn.preprocessing import PolynomialFeatures 
SS=StandardScaler()
X_train_scaled=SS.fit_transform(X_train)

SS=StandardScaler()
X_test_scaled=SS.fit_transform(X_test)

poly=PolynomialFeatures(degree=4)
X_train_poly=poly.fit_transform(X_train_scaled)

poly=PolynomialFeatures(degree=4)
X_test_poly=poly.fit_transform(X_test_scaled)

from sklearn.linear_model import Ridge 
modl =Ridge(alpha=.1)
modl.fit(X_train_poly,Y_train)
R_sq=modl.score(X_train_poly,Y_train)
print('coefficient of determination by Ridge Model for Polynomial feature is:', "{0:.3f}".format(R_sq))



poly=PolynomialFeatures(degree=5)
X_train_poly=poly.fit_transform(X_train_scaled)

poly=PolynomialFeatures(degree=5)
X_test_poly=poly.fit_transform(X_test_scaled)

from sklearn.linear_model import Ridge 
modl =Ridge(alpha=.1)
modl.fit(X_train_poly,Y_train)
R_sq=modl.score(X_train_poly,Y_train)
print('coefficient of determination by Ridge Model for Polynomial feature is:', "{0:.3f}".format(R_sq))






predict=pd.DataFrame()
predict['Predicted']=modl.predict(X_test_poly)
predict[0:10]
Y_test[0:10]

