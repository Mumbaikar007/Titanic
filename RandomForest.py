
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np


pd.options.mode.chained_assignment = None

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1

train["Embarked"] = train["Embarked"].fillna("S")
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

train["Age"] = train["Age"].fillna(train["Age"].median())

target = train["Survived"]

features_forest = train[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values
forest = RandomForestClassifier ( max_depth = 10, min_samples_split=2, n_estimators=100)
my_forest =forest.fit( features_forest, target)

test["Age"] = test["Age"].fillna(test["Age"].median())
test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"] = test["Embarked"].fillna("S")
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2


test.Fare[152] = test.Fare.median()
test_features = test[["Pclass", "Age", "Sex", "Fare", "SibSp", "Parch", "Embarked"]].values

my_predictions = my_forest.predict(test_features)


PassengerId = np.array ( test["PassengerId"] ).astype(int)
my_solution = pd.DataFrame ( my_predictions, PassengerId, columns=["Survived"])

my_solution.to_csv("Solution_RandomForest.csv", index_label=["PassengerId"])