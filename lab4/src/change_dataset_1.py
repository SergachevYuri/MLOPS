import pandas as pd


dataset = pd.from_csv('titanic.csv')

dataset = dataset[['PClass', 'Sex', 'Age']]
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