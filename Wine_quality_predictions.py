#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Combining the two datasets
d1 = pd.read_csv("winequality-red.csv", delimiter=";")
d2 = pd.read_csv("winequality-white.csv",delimiter=";")
d = pd.concat([d2,d1], ignore_index=True)
d


# In[3]:


d.head(10)


# In[4]:


d.describe()


# In[5]:


from matplotlib import pyplot as plt


# In[6]:


d["quality"].hist(rwidth=0.9, label="bars", color="k")
plt.xlabel("quality of wine")
plt.ylabel("Count of a quality")
plt.title("Histogram")
plt.legend()
plt.show()


# In[7]:


import seaborn as sns


# In[8]:


# Visualising the distribution of quality
sns.countplot(d["quality"])


# In[9]:


count5=d[d["quality"]==5]
count9=d[d["quality"]==9]
print(count5.shape,count9.shape)


# In[10]:


# Checking for outliers


# In[11]:


sns.boxplot(y=d["fixed acidity"])


# In[12]:


sns.boxplot(y=d["volatile acidity"])


# In[13]:


sns.boxplot(y=d["citric acid"])


# In[14]:


sns.boxplot(y=d["residual sugar"])


# In[15]:


sns.boxplot(y=d["chlorides"])


# In[16]:


sns.boxplot(y=d["free sulfur dioxide"])


# In[17]:


sns.boxplot(y=d["total sulfur dioxide"])


# In[18]:


sns.boxplot(y=d["density"])


# In[19]:


sns.boxplot(y=d["pH"])


# In[20]:


sns.boxplot(y=d["sulphates"])


# In[21]:


sns.boxplot(y=d["alcohol"])


# In[22]:


sns.boxplot(y=d["quality"])


# In[23]:


# Removing the outliers
def outliers(data, series):
    Q1 = data.sort_values([series])[series].quantile(0.25)
    Q3 = data.sort_values([series])[series].quantile(0.75)
    IQR = Q3 - Q1
    LB = Q1 - 1.5*IQR
    UB = Q3 + 1.5*IQR
    l = data.index[(data[series]<LB) | (data[series]>UB)]
    return l


# In[24]:


indexl = []
for series in ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]:
    ll=outliers(d,series)
    for i in range(len(ll)):
        indexl.append(ll[i]);


# In[25]:


print(indexl)
print(len(indexl))


# In[26]:


def delete(data, list):
    list = sorted(set(list))
    data = data.drop(list)
    return data


# In[27]:


d_removed = delete(d, indexl)
d_removed.shape


# In[28]:


# After removing the outliers, visualising the data


# In[29]:


sns.boxplot(y=d_removed["density"])


# In[30]:


sns.boxplot(y=d_removed["free sulfur dioxide"])


# In[31]:


sns.boxplot(y=d_removed["residual sugar"])


# In[32]:


#feature selection filter method
# Checking for correlation
l=d.drop("quality", axis=1)


# In[33]:


corr=l.corr()


# In[34]:


d.head(5)


# In[35]:


plt.figure(figsize=(20,12))
sns.heatmap(corr,annot=True)


# In[36]:


# As we can see there are no values closer to 1 or -1 so it is not a good idea to neglect any of the independent variables.


# In[37]:


X= d_removed.drop(columns=["quality"])
Y=d_removed["quality"]


# In[38]:


print(X.shape,Y.shape)


# In[39]:


from imblearn.over_sampling import SMOTE
smote=SMOTE(k_neighbors=4,random_state=4)
X, Y = smote.fit_resample(X, Y)


# In[40]:


#feature selection filter method
# Chi square test for feature selection process
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[41]:


bestfeatures = SelectKBest(score_func=chi2, k="all")
bestfit=bestfeatures.fit(X,Y)


# In[42]:


scores=pd.DataFrame(bestfit.scores_)
columns=pd.DataFrame(X.columns)
col=["scores","feature"]
df=pd.concat([scores, columns],axis=1)
df.columns=col


# In[43]:


df


# In[44]:


# density, pH - In this feature selection model density and pH performance is very low hence we may neglect them when we are building the model.


# In[45]:


Xb= d_removed.drop(columns=["quality","density","pH"])
Yb=d_removed["quality"]


# In[46]:


from imblearn.over_sampling import SMOTE
smote=SMOTE(k_neighbors=4,random_state=4)
X, Y = smote.fit_resample(Xb, Yb)


# In[47]:


from collections import Counter
print(Counter(Yb))
print(Counter(Y))


# In[48]:


# Training the model
from sklearn.model_selection import train_test_split


# In[49]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=4)


# In[50]:


# K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train, y_train)
ypredict=model.predict(x_test)
print("Accuracy: ", model.score(x_test,y_test))
pd.crosstab(y_test,ypredict)


# In[51]:


# Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train, y_train)
ypredict=model.predict(x_test)
print("Accuracy: ", model.score(x_test,y_test))
pd.crosstab(y_test,ypredict)


# In[52]:


# Random forest classifier
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)
ypredict=model.predict(x_test)
print("Accuracy: ", model.score(x_test,y_test))
pd.crosstab(y_test,ypredict)


# In[53]:


# Extra trees classifier
from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(x_train, y_train)
ypredict=model.predict(x_test)
print("Accuracy: ", model.score(x_test,y_test))
pd.crosstab(y_test,ypredict)


# In[ ]:




