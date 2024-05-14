import pandas as pd


dataset = pd.from_csv('titanic.csv')

dataset['Age'].fillna(dataset['Age'].mean(), inplace=True)
dataset.to_csv('titanic.csv')


'''
dvc add titanic_v2.csv
git add titanic_v2.csv.dvc .gitignore
git commit -m "Fill missing age values"
dvc push

'''