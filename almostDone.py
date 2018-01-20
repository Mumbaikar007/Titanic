
import pandas as pd
import numpy as np
from sklearn import tree

pd.options.mode.chained_assignment = None

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

'''
print( train.head() )
print( test.head() )
'''


train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1


train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

train["Age"] = train["Age"].fillna ( train["Age"].median())


target = train["Survived"].values
features_one = train[["Pclass", "Sex", "Age", "Fare"]]


my_tree_one = tree.DecisionTreeClassifier()
my_tree_one = my_tree_one.fit ( features_one, target)


test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1

test.Fare[152] = test.Fare.median()
test_features = test[["Pclass", "Sex", "Age", "Fare"]].values

my_predictions = my_tree_one.predict(test_features)


PassengerId = np.array ( test["PassengerId"] ).astype(int)
my_solution = pd.DataFrame ( my_predictions, PassengerId, columns=["Survived"])

my_solution.to_csv("Solution_almostDone.csv", index_label=["PassengerId"])