import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

# Загрузка данных датасета ирисов
def load_and_preprocess_data():
    iris = load_iris()
    X = pd.DataFrame(iris.data, columns=iris.feature_names)
    y = pd.DataFrame(iris.target, columns=['target'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Сохраняем скачанные данные
def save_datasets(X_train, X_test, y_train, y_test):
    X_train.to_csv('X_train.csv', index=False)
    X_test.to_csv('X_test.csv', index=False)
    y_train.to_csv('y_train.csv', index=False)
    y_test.to_csv('y_test.csv', index=False)



X_train, X_test, y_train, y_test = load_and_preprocess_data()
save_datasets(X_train, X_test, y_train, y_test)