import pandas as pd


dataset = pd.read_csv('titanic.csv')

dataset = dataset[['Pclass', 'Sex', 'Age']]
dataset.to_csv('titanic.csv')


'''
Добавляем файл в DVC и делаем коммит в Git
git add .
git commit -m "Initial titanic dataset"
dvc add titanic_v1.csv
git add titanic_v1.csv.dvc .gitignore
git commit -m "Add titanic dataset to DVC"
dvc push
'''