from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

def preprocess_data(file_path):
    data = pd.read_csv(file_path)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data[['temperature']])
    data['temperature'] = scaled_data
    return data

for folder in ['data']:
    for file_name in os.listdir(folder):
        file_path = f'{folder}/{file_name}'
        data = preprocess_data(file_path)
        data.to_csv(file_path, index=False)
