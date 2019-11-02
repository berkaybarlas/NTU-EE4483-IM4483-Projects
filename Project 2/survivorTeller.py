# linear algebra
import numpy as np 

# data processing
import pandas as pd 

# data visualization
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import style

# Algorithms
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

# exec("%matplotlib inline")

test_df = pd.read_csv("test.csv")
train_df = pd.read_csv("train.csv")

# print("---Train Data Info---")
# train_df.info()
# print("---Train Data Description---")
# print(train_df.describe())
# print("test")

total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
print(missing_data.head(11))

survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

# Age Graph for Women
women = train_df[train_df['Sex']=='female']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
# Age Graph for Men
men = train_df[train_df['Sex']=='male']
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
ax.set_title('Male')
plt.show()

# The effect of ticket class')
tclass = sns.barplot(x='Pclass', y='Survived', data=train_df)
tclass.legend()
tclass.set_title('Ticket Classes')
plt.show()

data = [train_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']

plt.close()  
fig, rAx = plt.subplots(3,1)
# The effect of siblings
sns.factorplot('SibSp','Survived', data=train_df, aspect = 2.5, ax=rAx[0] )
plt.close()  
# The effect of parents
sns.factorplot('Parch','Survived', data=train_df, aspect = 2.5, ax=rAx[1])
plt.close()                      
# The effect of relatives   
sns.factorplot('relatives','Survived', data=train_df, aspect = 2.5, ax=rAx[2])
plt.close()

plt.show()
