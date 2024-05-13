import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error
import os

model = joblib.load('linear_regression_model.pkl')
test_data = pd.DataFrame()

for file_name in os.listdir('test'):
    data = pd.read_csv(f'test/{file_name}')
    test_data = pd.concat([test_data, data])

X_test = test_data[['day']]
y_test = test_data['temperature']
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

print(f'Mean Squared Error: {mse}')
