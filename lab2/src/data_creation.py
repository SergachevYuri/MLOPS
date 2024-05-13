import numpy as np
import pandas as pd
import os

def generate_data(num_points, noise_level=0.0, anomaly=False):
    x = np.linspace(0, num_days-1, num_days, dtype=int)
    y = 10 + 15 * np.sin(2 * np.pi * x / 365)  # базовая годовая температурная модель
    z = np.random.normal(750, 50, num_points) # базовая годовая модель давления
    noise = noise_level * np.random.normal(size=num_points)
    if anomaly:
        y[np.random.randint(0, num_points, size=5)] += 20  # добавление аномалий в случайные дни
    return pd.DataFrame({'day': x, 'temperature': y + noise, 'pressure': z})

np.random.seed(42)
num_days = 365

# Создание тренировочных данных
if not os.path.exists('data'):
    os.makedirs('data')
train_data1 = generate_data(num_days, noise_level=1.0)
train_data1.to_csv('data/train_data.csv', index=False)


test_data1 = generate_data(num_days, noise_level=1.0)
test_data1.to_csv('data/test_data.csv', index=False)