import pandas as pd
from catboost.datasets import titanic


titanic, _ = titanic()
titanic.to_csv('titanic.csv')