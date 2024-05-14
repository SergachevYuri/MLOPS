import pandas as pd


dataset = pd.read_csv('titanic.csv')
dataset = pd.get_dummies(dataset, columns=['Sex'])
dataset.to_csv('titanic.csv', index=False)