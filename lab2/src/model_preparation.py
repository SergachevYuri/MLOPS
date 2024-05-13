from sklearn.linear_model import LinearRegression
import pandas as pd
import os
import joblib

model = LinearRegression()
train_data = pd.DataFrame()

data = pd.read_csv('data/train_data.csv')
train_data = pd.concat([train_data, data])

X_train = train_data[['day']]
y_train = train_data['temperature']
model.fit(X_train, y_train)

if not os.path.exists('model'):
    os.makedirs('model')

joblib.dump(model, 'model/linear_regression_model.pkl')
