from sklearn.linear_model import LinearRegression
import pandas as pd
import os
import joblib

model = LinearRegression()
train_data = pd.DataFrame()

for file_name in os.listdir('train'):
    data = pd.read_csv(f'train/{file_name}')
    train_data = pd.concat([train_data, data])

X_train = train_data[['day']]
y_train = train_data['temperature']
model.fit(X_train, y_train)

joblib.dump(model, 'linear_regression_model.pkl')
