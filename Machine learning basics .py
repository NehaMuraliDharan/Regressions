#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn.preprocessing as Imputer 


# In[64]:


dataset=pd.read_csv('_Data.csv')


# In[65]:


print(dataset)


# In[66]:


del dataset['Unnamed: 4']
del dataset['Unnamed: 5']


# In[67]:


print(dataset)
print(type(dataset))


# In[68]:


X=dataset.iloc[:, :-1].values
y=dataset.iloc[:,3].values
print(X)
print(y)
X_1=pd.DataFrame(X)
y_1=pd.DataFrame(y)
print(X_1)
print(y_1)


# In[69]:


print(dataset)


# In[70]:


from sklearn.impute import SimpleImputer
imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean',)
imp_mean=imp_mean.fit(X[:,1:3])
imp_mean=imp_mean.transform(X[:,1:3])


# In[71]:


imp=pd.DataFrame(imp_mean)
print(imp)


# In[72]:


X=pd.DataFrame(X)
print(X)


# In[73]:


X[1]=imp[0]
X[2]=imp[1]
print(X)


# In[74]:


X=np.array(X)
print(X)


# In[75]:


from sklearn.preprocessing import LabelEncoder 


# In[76]:


labelencoder_X= LabelEncoder()
X[:,0]=labelencoder_X.fit_transform(X[:,0])


# In[77]:


print(X)


# In[78]:


from sklearn.preprocessing import OneHotEncoder


# In[79]:


print(X)


# In[80]:


X_2=X[:,0]
X_2=pd.DataFrame(X_2)
print(X_2)


# In[81]:


onehotencoder=OneHotEncoder(categories='auto', sparse=False)
X_2=onehotencoder.fit_transform(X_2)


# In[82]:


print(X_2)


# In[83]:


print(X)


# In[84]:


X=pd.DataFrame(X)


# In[85]:


print(X)


# In[86]:


X.insert(loc=2, column='dkckw', value=X_2[:,2])


# In[87]:


np.array(X)


# In[88]:


X=pd.DataFrame(X)
print(X)


# In[89]:


X.to_string(header=False)


# In[90]:


X


# In[91]:


X = X.rename(columns={'2': '0', '1': '1', 'dkckw': '2', '0': '3', '1': '4', '2': '5' })
print(X)


# In[ ]:





# In[92]:


print(X)


# In[ ]:





# In[ ]:





# In[93]:


labelencoder=LabelEncoder()
y=labelencoder.fit_transform(y)


# In[94]:


print(y)


# In[95]:


from sklearn.model_selection import train_test_split


# In[96]:


X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.2,random_state=0)


# In[97]:


print(X_train)
print(X_test)
print(y_train)
print(y_test)


# In[98]:


print(X)


# In[99]:


from sklearn.preprocessing import StandardScaler 


# In[100]:


sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)


# In[101]:


print(X_train)
print('---------------------------------------------------------------------------')
print(X_test)


# In[102]:


dataset_1=pd.read_csv('Simple_Linear_Regression_Salary_Data.csv')


# In[103]:


print(dataset_1)


# In[104]:


X=dataset_1.iloc[:,:-1].values 
y=dataset_1.iloc[:,1].values 


# In[105]:


print(X)
print('------------------------------------------------------------------')
print(y)


# In[106]:


from sklearn.model_selection import train_test_split


# In[107]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=0)


# In[108]:


print(X_train)
print('------------------------------')
print(X_test)
print('------------------------------')
print(y_train)
print('------------------------------')
print(y_test)


# In[109]:


from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[110]:


y_pred = regressor.predict(X_test)


# In[111]:


plt.scatter(X_train,y_train, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Training set')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()


# In[112]:


plt.scatter(X_test,y_test, color='red')
plt.plot(X_train,regressor.predict(X_train),color='blue')
plt.title('Training set')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()


# In[113]:


dataset_2=pd.read_csv('Multiple_Linear_Regression_50_Startups.csv')


# In[114]:


print(dataset_2)


# In[115]:


X=dataset_2.iloc[:,:-1].values
y=dataset_2.iloc[:,4].values


# In[116]:


print(X)


# In[117]:


print(y)


# In[121]:


from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
X[:,3]=labelencoder.fit_transform(X[:,3])


# In[144]:


x=(X[:,3])
print(x)


# In[159]:


x=pd.DataFrame(x)
X=pd.DataFrame(X)


# In[147]:


from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categories='auto', sparse=False)
x=onehotencoder.fit_transform(x)
print(x)


# In[157]:


x=pd.DataFrame(x)
print(x)


# In[169]:


X.insert(loc=0, column='C', value=x[1])


# In[170]:


print(X)


# In[173]:


del X[3]


# In[174]:


print(X)


# In[182]:


y=dataset_2.iloc[:,4].values
y
X=np.array(X)


# In[183]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.2, random_state=0)


# In[188]:


y_train=pd.DataFrame(y_train)


# In[192]:


X_train


# In[193]:


X_test


# In[194]:


y_test


# In[195]:


y_train


# In[200]:


X_train


# In[201]:


X_test 


# In[204]:


from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,y_train)


# In[205]:


y_pred=regressor.predict(X_test)


# In[208]:


y_pred


# In[209]:


y_test


# In[210]:


X


# In[211]:


X=pd.DataFrame(X)


# In[213]:


X


# In[214]:


del X[0]


# In[215]:


X=np.array(X)


# In[218]:


import statsmodels.formula.api as sm 
X=np.append(arr=np.ones((50,1)).astype(int), values=X, axis=1)


# In[219]:


X


# In[ ]:





# In[232]:


import statsmodels.api as sm
X_opt=X[:,[0,1,2,3,4,5]]
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# In[233]:


X_opt = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# In[229]:


X_opt = np.array(X[:, [0, 3, 4, 5]], dtype=float)
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# In[234]:



X_opt = np.array(X[:, [0, 3, 5]], dtype=float)
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# In[235]:


X_opt = np.array(X[:, [0, 3]], dtype=float)
regressor_OLS=sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()


# In[237]:


X_new=X[:,[0,3]]


# In[240]:


X_new=pd.DataFrame(X_new)


# In[241]:


X_new


# In[273]:


import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 


# In[6]:


dataset_3=pd.read_csv('Polynomial_Regression_Position_Salaries.csv')


# In[7]:


X=dataset_3.iloc[:,1:2].values
y=dataset_3.iloc[:,2].values


# In[8]:


X


# In[9]:


from sklearn.linear_model import LinearRegression


# In[10]:


lin_reg=LinearRegression()
lin_reg.fit(X,y)


# In[11]:


from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4)
X_poly=poly_reg.fit_transform(X)


# In[12]:


lin_reg2=LinearRegression()
lin_reg2.fit(X_poly,y)


# In[13]:


plt.scatter(X,y, color='red')
plt.plot(X,lin_reg.predict(X),color='blue')
plt.title('truth or bluff')
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()


# In[14]:


plt.scatter(X,y, color='red')
plt.plot(X,lin_reg2.predict(X_poly),color='blue')
plt.title('truth or bluff')
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()


# In[15]:


X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape((len(X_grid), 1))


# In[16]:


plt.scatter(X,y, color='red')
plt.plot(X_grid,lin_reg2.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('truth or bluff')
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()


# In[2]:


import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 


# In[ ]:





# In[30]:


dataset=pd.read_csv('Polynomial_Regression_Position_Salaries.csv')


# In[31]:


X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


# In[32]:


from sklearn.svm import SVR
regressor =SVR(kernel='rbf')
regressor.fit(X,y)


# In[37]:


plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Buff')
plt.xlabel("Position level")
plt.ylabel("Salary")


# In[36]:


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)


# In[35]:


X=pd.DataFrame(X)
y=pd.DataFrame(y)


# In[38]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[39]:


dataset=pd.read_csv('Polynomial_Regression_Position_Salaries.csv')


# In[43]:


X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values


# In[44]:


X=pd.DataFrame(X)
y=pd.DataFrame(y)


# In[45]:


from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
sc_y=StandardScaler()
X=sc_X.fit_transform(X)
y=sc_y.fit_transform(y)


# In[47]:


from sklearn.svm import SVR
regressor =SVR(kernel='rbf')
regressor.fit(X,y)


# In[48]:


plt.scatter(X,y,color='red')
plt.plot(X,regressor.predict(X), color="blue")
plt.title('Truth or Bluff')


# In[2]:


import pandas as pd


# In[5]:


dataset_4=pd.read_csv('Decision_Tree_Regression_Position_Salaries.csv')


# In[6]:


dataset_4


# In[8]:


X=dataset_4.iloc[:,1:2].values
y=dataset_4.iloc[:,2].values


# In[9]:


import matplotlib.pyplot as plt


# In[10]:


plt.scatter(X,y,color="red")


# In[28]:


from sklearn.tree import DecisionTreeRegressor 
regressor= DecisionTreeRegressor(random_state=0)
regressor.fit (X,y)


# In[29]:


plt.scatter(X,y,color="red")
plt.plot(X,regressor.predict(X),color='blue')
plt.title('Position Vs. salary')
plt.xlabel('Position')
plt.ylabel('salary')
plt.show()


# In[30]:


import numpy as np


# In[34]:


X_grid=np.arange(min(X), max(X), 0.01)


# In[35]:


X_grid=X_grid.reshape((len(X_grid), 1))


# In[36]:


plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid), color="blue")
plt.title("Position Vs. Salary")
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


# In[37]:


import pandas as pd 


# In[38]:


dataset_5=pd.read_csv('Random_Forest_Regression_Position_Salaries.csv')


# In[39]:


dataset_5


# In[40]:


X=dataset_5.iloc[:,1:2].values
y=dataset_5.iloc[:,2].values 


# In[42]:


from sklearn.ensemble import RandomForestRegressor 


# In[51]:


regressor=RandomForestRegressor(random_state=0, n_estimators=300)
regressor.fit(X,y)


# In[52]:


plt.scatter(X,y,color='red')
plt.plot(X, regressor.predict(X),color='blue')


# In[53]:


X_grid=np.arange(min(X), max(X), 0.01)
X_grid=X_grid.reshape((len(X_grid),1))


# In[54]:


plt.scatter(X,y,color='red')
plt.plot(X_grid, regressor.predict(X_grid),color='blue')


# In[57]:


y_pred=regressor.predict([[6.5]])


# In[58]:


y_pred


# In[36]:


import pandas as pd 


# In[37]:


daatset_5=pd.read_csv('titanic.csv')


# In[38]:


daatset_5


# In[39]:


X=daatset_5.iloc[:,[4,2]].values 
y=daatset_5.iloc[:,1].values


# In[40]:


X


# In[41]:


y


# In[42]:


from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# In[43]:


labelencoder_X = LabelEncoder()

X[:, 0] = labelencoder_X.fit_transform(X[:, 0])


# In[44]:


X


# In[47]:


import numpy as np
onehotencoder = OneHotEncoder(categories='auto', sparse=False)

X = np.array(onehotencoder.fit_transform(X))


# In[48]:


X


# In[49]:


X=pd.DataFrame(X)


# In[50]:


X


# In[51]:


del X[4]


# In[52]:


del X[3]


# In[53]:


del X[2]


# In[54]:


print(X)


# In[56]:


x=daatset_5.iloc[:,2].values


# In[57]:


print(x)


# In[58]:


x=pd.DataFrame(x)


# In[59]:


x


# In[60]:


X


# In[61]:


X[2]=x[0]


# In[62]:


print(X)


# In[63]:


from sklearn.model_selection import train_test_split


# In[64]:


X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=0,test_size=0.25)


# In[65]:


X_train


# In[66]:


X_test


# In[68]:


from sklearn.preprocessing import StandardScaler


# In[69]:


sc=StandardScaler()


# In[71]:


X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[72]:


X_train


# In[73]:


X_test


# In[74]:


from sklearn.linear_model import LogisticRegression


# In[77]:



classifier = LogisticRegression(random_state = 0)

classifier.fit(X_train, y_train)


# In[78]:


y_pred=classifier.predict(X_test)


# In[79]:


from sklearn.metrics import confusion_matrix


# In[80]:


cm=confusion_matrix(y_test,y_pred)


# In[81]:


print(cm)


# In[82]:


139+84


# In[83]:


139/223


# In[84]:


import pandas as pd 


# In[85]:


dataset_6=pd.read_csv('Logistic_Regression_Social_Network_Ads.csv')


# In[86]:


print(dataset_6)


# In[87]:


X=dataset_6.iloc[:,[1,2,3]].values
y=dataset_6.loc[:4].values


# In[88]:


from sklearn.preprocessing import LabelEncoder 


# In[89]:


print(X)


# In[91]:


labelencoder_X = LabelEncoder()

X[:, 0] = labelencoder_X.fit_transform(X[:, 0])


# In[99]:


print(X)


# In[100]:


x=X[0]
x=pd.DataFrame(x)


# In[102]:


print(x)


# In[103]:


import pandas as pd 


# In[105]:


dataset_7=pd.read_csv('K_Nearest_Neighbors_Social_Network_Ads.csv')


# In[107]:


dataset_7


# In[110]:


x=dataset_7['Gender']


# In[111]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder


# In[113]:


labelencoder=LabelEncoder()
x=labelencoder.fit_transform(x)


# In[121]:


x=pd.DataFrame(x)


# In[122]:


onehotencoder=OneHotEncoder(categories='auto', sparse=False)
x=onehotencoder.fit_transform(x)


# In[125]:


x


# In[126]:


import numpy as np


# In[127]:


x=np.array(x)


# In[135]:


X = dataset_7.iloc[:, [2, 3]].values
y=dataset_7.iloc[:,4].values


# In[136]:


X=np.append(x, values=X, axis=1)


# In[137]:


X


# In[138]:


from sklearn.model_selection import train_test_split


# In[142]:


X_train,X_test,y_train,y_test=train_test_split(X, y, test_size=0.25, random_state=0)


# In[144]:


from sklearn.preprocessing import StandardScaler


# In[145]:


sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[146]:


from sklearn.neighbors import KNeighborsClassifier


# In[149]:


classifier=KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
classifier.fit(X_train,y_train)


# In[150]:


y_pred= classifier.predict(X_test)


# In[151]:


from sklearn.metrics import confusion_matrix


# In[152]:


cm= confusion_matrix(y_pred, y_test)


# In[153]:


cm 


# In[5]:


import pandas as pd


# In[6]:


dataset_8= pd.read_csv('SVM_Social_Network_Ads.csv')


# In[7]:


dataset_8


# In[8]:


X=dataset_8.iloc[:,1:4].values
y=dataset_8.iloc[:,4].values


# In[9]:


X=pd.DataFrame(X)
y=pd.DataFrame(y)
print(X)
print(y)


# In[10]:


x=X[0]


# In[11]:


print(x)


# In[12]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder()
x=le.fit_transform(x)


# In[13]:


x=pd.DataFrame(x)
print(x)


# In[14]:


ohe=OneHotEncoder(categories='auto', sparse=False)
x=ohe.fit_transform(x)


# In[15]:


x


# In[16]:


X


# In[17]:


del X[0]


# In[18]:


X


# In[19]:


import numpy as np


# In[20]:


X=np.array(X)
x=np.array(x)


# In[21]:


X


# In[22]:


x


# In[23]:


X=np.append(x, values=X, axis=1)


# In[24]:


X


# In[25]:


from sklearn.model_selection import train_test_split


# In[26]:


X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.25, random_state=0)


# In[27]:


from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[29]:


from sklearn.svm import SVC


# In[30]:


classifier=SVC(random_state=0, kernel='rbf')
classifier.fit(X_train,y_train)


# In[60]:


Y=classifier.predict(X_test)


# In[61]:


import pandas as pd


# In[62]:


dataset_9=pd.read_csv('titanic.csv')


# In[94]:


dataset_8


# In[100]:


X=dataset_8.iloc[:,[2,3]].values 
X=pd.DataFrame(X)
y=dataset_8.iloc[:,-1].values 
y=pd.DataFrame(y)


# In[103]:


x=dataset_8.iloc[:,1]
x=pd.DataFrame(x)


# In[ ]:





# In[104]:


x


# In[105]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder()
x=le.fit_transform(x)


# In[106]:


x


# In[107]:


x=pd.DataFrame(x)


# In[108]:


ohe=OneHotEncoder(categories='auto', sparse=False)
x=ohe.fit_transform(x)


# In[109]:


x


# In[110]:


import numpy as np
x=np.array(x)
X=np.array(X)


# In[111]:


X=np.append(x, values=X, axis=1)


# In[112]:


from sklearn.model_selection import train_test_split


# In[113]:


X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.25, random_state=0)


# In[114]:


from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[115]:


from sklearn.naive_bayes import GaussianNB


# In[149]:


classifier=GaussianNB()
classifier.fit(X_train, y_train)


# In[150]:


import pandas as pd
dataset_8= pd.read_csv('SVM_Social_Network_Ads.csv')


# In[151]:


dataset_8


# In[152]:


X=dataset_8.iloc[:,1:4].values
y=dataset_8.iloc[:,4].values


# In[153]:


X=pd.DataFrame(X)
y=pd.DataFrame(y)
print(X)
print(y)


# In[154]:


x=X[0]
print(x)


# In[155]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
le=LabelEncoder()
x=le.fit_transform(x)


# In[156]:


x=pd.DataFrame(x)
print(x)


# In[157]:


ohe=OneHotEncoder(categories='auto', sparse=False)
x=ohe.fit_transform(x)


# In[158]:


X=np.array(X)


# In[159]:


x=np.array(x)


# In[160]:


X=np.append(x, values=X, axis=1)


# In[168]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.25, random_state=0)


# In[169]:


X


# In[163]:


X=pd.DataFrame(X)


# In[164]:


X


# In[165]:


del X[2]


# In[166]:



X


# In[170]:


from sklearn.preprocessing import StandardScaler 
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[171]:


from sklearn.ensemble import RandomForestClassifier 


# In[177]:


classifier=RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
classifier.fit(X_train, y_train)


# In[178]:


y_pred=classifier.predict(X_test)


# In[179]:


from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_pred, y_test)


# In[180]:


cm 


# In[1]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 


# In[2]:


dataset=pd.read_csv('Mall_Customers.csv')


# In[3]:


dataset


# In[4]:


X=dataset.iloc[:,[3,4]]


# In[5]:


X


# In[6]:


from sklearn.cluster import KMeans


# In[9]:


wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[14]:


kmeans=KMeans(n_clusters=5, n_init=10, init='k-means++', max_iter=300)
y_kmeans=kmeans.fit_predict(X)
print(y_kmeans)


# In[18]:


y_kmeans=pd.DataFrame(y_kmeans)


# In[19]:


y_kmeans


# In[28]:


X


# In[30]:


plt.scatter(X['Annual Income (k$)'], X['Spending Score (1-100)'], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);


# In[31]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 


# In[32]:


dataset 


# In[33]:


X=dataset.iloc[:,[3,4]]


# In[35]:


import scipy.cluster.hierarchy as sch


# In[37]:


dendrogram=sch.dendrogram(sch.linkage(X, method='ward'))
plt.title('dendogram')
plt.xlabel('Annual income')
plt.xlabel('Spending Score')
plt.show()


# In[38]:


from sklearn.cluster import AgglomerativeClustering 


# In[46]:


hc=AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')


# In[47]:


y_hc=hc.fit_predict(X)


# In[10]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


# In[11]:


dataset= pd.read_csv('Apriori_Python_Market_Basket_Optimisation.csv')


# In[12]:


transactions=[]
for i in range(0,7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0,20)])


# In[ ]:


transactions


# In[ ]:





# In[ ]:


from apyori import apriori


# In[ ]:


rules=apriori(transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2 )


# In[ ]:


results=list(rules)


# In[2]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# In[1]:


print('hey')


# In[2]:


print('is this working?')


# In[3]:





# In[ ]:




