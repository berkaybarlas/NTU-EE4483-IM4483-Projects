import warnings
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

from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV

# Ignore Pandas warnings
warnings.filterwarnings('ignore')

# exec("%matplotlib inline")

test_df = pd.read_csv("test.csv")
test_df_sol = pd.read_csv("test-solution.csv")
train_df = pd.read_csv("train.csv")
data = [train_df, test_df]
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
#plt.show()

# The effect of ticket class')
tclass = sns.barplot(x='Pclass', y='Survived', data=train_df)
tclass.legend()
tclass.set_title('Ticket Classes')
#plt.show()

for dataset in data:
    dataset['Relatives'] = dataset['SibSp'] + dataset['Parch']

plt.close()  
fig, rAx = plt.subplots(3,1)
# The effect of siblings
sns.factorplot('SibSp','Survived', data=train_df, aspect = 2.5, ax=rAx[0] )
plt.close()

# The effect of parents
sns.factorplot('Parch','Survived', data=train_df, aspect = 2.5, ax=rAx[1])
plt.close()

# The effect of Relatives   
sns.factorplot('Relatives','Survived', data=train_df, aspect = 2.5, ax=rAx[2])
plt.close()

#plt.show()

# Use name to get titles and drop it
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5 ,"Rare": 6}

for dataset in data:
    # extract titles
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
    # replace titles with a more common title or as Rare
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Capt', 'Col',
                                            'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    # convert titles into numbers
    dataset['Title'] = dataset['Title'].map(titles)
    # filling NaN with 0, to get safe
    dataset['Title'] = dataset['Title'].fillna(0)
    
    dataset = dataset.drop(['Name'], axis=1)

#Drop the Name feature
data[0] = data[0].drop(['Name'], axis=1)
data[1] = data[1].drop(['Name'], axis=1)

#Drop the PassengerId feature
data[0] = data[0].drop(['PassengerId'], axis=1)

# Drop the Cabin and Ticket feature
data[0] = data[0].drop(['Cabin'], axis=1)
data[1] = data[1].drop(['Cabin'], axis=1)

data[0] = data[0].drop(['Ticket'], axis=1)
data[1] = data[1].drop(['Ticket'], axis=1)

data[0] = data[0].drop(['Parch'], axis=1)
data[1] = data[1].drop(['Parch'], axis=1)

data[0] = data[0].drop(['SibSp'], axis=1)
data[1] = data[1].drop(['SibSp'], axis=1)

# Convert Fare  from float to int64
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype(int)

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

#Fill Missing Data in Age
#data[0]['Age'].fillna(lambda x: int(groupMean[x]), inplace = True)

for dataset in data:
    dataset['Age'] = dataset.groupby(['Fare','Sex'])['Age'].transform(lambda x: x.fillna(x.mean()))
    dataset['Age'] = dataset['Age'].astype(int)
# Fill missing embarktion data
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna('S')

# Convert Features

# Convert Sex and Ports to int
ports = {"S": 0, "C": 1, "Q": 2}
genders = {"male": 0, "female": 1}
for dataset in data:
    dataset['Sex'] = dataset['Sex'].map(genders)
    dataset['Embarked'] = dataset['Embarked'].map(ports)

# Convert Age data to evenly distributed groups
for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[ dataset['Age'] <= 5, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 5) & (dataset['Age'] <= 11), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 3
    #dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 35), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 35) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 7

for dataset in data:
    dataset['Age_Class']= dataset['Age'] * dataset['Pclass']
#print(data[0].head(5))
#print(data[1].head(5))

X_train = data[0].drop("Survived", axis=1)
Y_train = data[0]["Survived"]
X_test = data[1].drop("PassengerId", axis=1).copy()
Y_test = test_df_sol.Survived
print(X_train.head())

### Training Models ###

# Random Forest 
random_forest = RandomForestClassifier(n_estimators=100, 
                                       oob_score=True, 
                                       random_state=0)
random_forest.fit(X_train, Y_train)
rf_Y_prediction = random_forest.predict(X_test)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 3)

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(X_train, Y_train)
log_Y_prediction = logreg.predict(X_test)
acc_log = round(logreg.score(X_train, Y_train) * 100, 3)

# K-mean
kNN = KNeighborsClassifier(n_neighbors = 2)
kNN.fit(X_train, Y_train)
kNN_Y_prediction = kNN.predict(X_test)
acc_kNN = round(kNN.score(X_train, Y_train) * 100, 3)

# Decision Tree
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
dt_Y_prediction = decision_tree.predict(X_test)
acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 3)

results = pd.DataFrame({
    'Model':['KNN', 'Logistic Regression', 
              'Random Forest', 'Decision Tree'],
    'Score': [acc_kNN, acc_log, acc_random_forest, acc_decision_tree]})
result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
print(result_df)

importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(random_forest.feature_importances_,3)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')
print(importances.head(15))

ids = data[1]['PassengerId']
rf_output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': rf_Y_prediction })
knn_output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': kNN_Y_prediction })
decision_tree_output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': dt_Y_prediction })
rf_output.to_csv('rf-titanic-predictions.csv', index = False)
knn_output.to_csv('knn-titanic-predictions.csv', index = False)

rf_acc = accuracy_score(Y_test, rf_Y_prediction) * 100
knn_acc = accuracy_score(Y_test, kNN_Y_prediction) * 100
decision_acc = accuracy_score(Y_test, dt_Y_prediction) * 100
print('Random forest accuracy: % 5.2f' %(rf_acc))
print('kNN accuracy: % 5.2f' %(knn_acc))
print('Decision tree accuracy: % 5.2f' %(decision_acc))

# Choose the type of classifier. 
clf = RandomForestClassifier()

# Choose some parameter combinations to try
parameters = {'n_estimators': [50,100],
              'oob_score': [True],
              'random_state': [0],
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 4, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': [1,5,]
             }
# Type of scoring used to compare parameter combinations
acc_scorer = make_scorer(accuracy_score)

# Run the grid search
grid_obj = GridSearchCV(clf, parameters, scoring=acc_scorer)
grid_obj = grid_obj.fit(X_train, Y_train)

# Set the clf to the best combination of parameters
clf = grid_obj.best_estimator_

# Fit the best algorithm to the data. 
clf.fit(X_train, Y_train)

predictions = clf.predict(X_test)
fitted_rf_acc = accuracy_score(Y_test, predictions) * 100
print('Best Random tree accuracy: % 5.2f' %(fitted_rf_acc))
rf_output = pd.DataFrame({ 'PassengerId' : ids, 'Survived': rf_Y_prediction })
rf_output.to_csv('best-rf-titanic-predictions.csv', index = False)

results = pd.DataFrame(data[1], {'Survived': rf_Y_prediction })
results.to_csv('results.csv', index = False)