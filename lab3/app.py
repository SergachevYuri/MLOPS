import gradio as gr
from joblib import load
import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

# Загрузка модели и данных ириса для получения названий классов
model = load('iris_model.joblib')
iris = load_iris()
iris_classes = iris.target_names

def predict(sepal_length, sepal_width, petal_length, petal_width):
    data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]],
                        columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
    prediction = model.predict(data)
    # Возвращаем название класса вместо индекса
    return iris_classes[prediction[0]]

iface = gr.Interface(fn=predict,
                     inputs=["number", "number", "number", "number"],
                     outputs="text",  # Изменено на 'text' для вывода названий
                     title="Предсказание ириса",
                     description="Введите данные ириса для предсказания класса.")
if __name__ == "__main__":
    iface.launch(server_name='0.0.0.0', server_port=7860)
