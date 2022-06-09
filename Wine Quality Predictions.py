#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[2]:


# Combining the two datasets given
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


d["quality"].hist(rwidth=0.9)
plt.xlabel("quality of wine")
plt.ylabel("Count of a quality")
plt.title("Histogram")
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
def outliers(data, feature):
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    LB = Q1 - 1.5*IQR
    UB = Q3 + 1.5*IQR
    l = data.index[(data[feature]<LB) | (data[feature]>UB)]
    return l


# In[24]:


indexl = []
for feature in ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]:
    indexl.extend(outliers(d, feature))


# In[25]:


print(indexl)
print(len(indexl))


# In[26]:


def remove(data, list):
    list = sorted(set(list))
    data = data.drop(list)
    return data


# In[27]:


d_removed = remove(d, indexl)
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


# Checking for correlation


# In[33]:


corr=d.corr()


# In[34]:


fig, ax = plt.subplots(figsize=(20,12))
sns.heatmap(corr, annot=True)


# In[35]:


# As we can see there are no values closer to 1 or -1 so it is not a good idea to neglect any of the independent variables.


# In[36]:


l = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"]
for i in range(len(l)):
    l1=[l[i]]
    X= d_removed[l1]
    Y=d_removed["quality"]
    from imblearn.over_sampling import SMOTE
    smote=SMOTE(k_neighbors=4,random_state=4)
    X, Y = smote.fit_resample(X, Y)
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20)
    from sklearn.ensemble import RandomForestClassifier
    model=RandomForestClassifier()
    model.fit(x_train, y_train)
    print("Accuracy: ", model.score(x_test,y_test))
    if i!= len(l)-1 :
        l1=l1.append(l[i+1])
    else:
        break
print(Y.value_counts())


# In[37]:


# pH, total sulfur dioxide - when they are added into the list then the performance of model is decreasing hence we may neglect them when building our model


# In[38]:


X= d_removed.drop(columns=["quality"])
Y=d_removed["quality"]


# In[39]:


print(X.shape,Y.shape)


# In[40]:


from imblearn.over_sampling import SMOTE
smote=SMOTE(k_neighbors=4,random_state=4)
X, Y = smote.fit_resample(X, Y)


# In[41]:


# Chi square test for feature selection process
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


# In[42]:


bestfeatures = SelectKBest(score_func=chi2, k="all")
bestfit=bestfeatures.fit(X,Y)


# In[43]:


scores=pd.DataFrame(bestfit.scores_)
columns=pd.DataFrame(X.columns)
col=["scores","feature"]
df=pd.concat([scores, columns],axis=1)
df.columns=col


# In[44]:


df


# In[45]:


# density, pH - In this feature selection model density and pH performance is very low hence we may neglect them when we are building the model.


# In[46]:


X= d_removed.drop(columns=["quality","density","pH","total sulfur dioxide"])
Y=d_removed["quality"]


# In[47]:


from imblearn.over_sampling import SMOTE
smote=SMOTE(k_neighbors=4,random_state=4)
X, Y = smote.fit_resample(X, Y)


# In[48]:


# Training the model
from sklearn.model_selection import train_test_split


# In[49]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=100)


# In[54]:


# K Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier()
model.fit(x_train, y_train)
print("Accuracy: ", model.score(x_test,y_test))


# In[55]:


# Decision tree classifier
from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(x_train, y_train)
print("Accuracy: ", model.score(x_test,y_test))


# In[56]:


# Random forest classifier
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)
print("Accuracy: ", model.score(x_test,y_test))


# In[59]:


# Extra trees classifier
from sklearn.ensemble import ExtraTreesClassifier
model=ExtraTreesClassifier()
model.fit(x_train, y_train)
print("Accuracy: ", model.score(x_test,y_test))


# In[ ]:





# In[ ]:




