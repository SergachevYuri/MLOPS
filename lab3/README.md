# MLops HW 3

Данный проект представляет собой использование Gradio для использования предобучченной модели машинного обучения,, которая обучается на моменте подключения докер контейнера.

## preprocess.py
Данный скрипит для скачивания и сохранения данных датасета ирисов
Создаются файлы
- X_train.csv
- X_test.csv
- y_train.csv
- y_test.csv

## train.py
Данный скрипт обучает и сохраняет модель Логистической Регресии
- iris_model.joblib

## app.py
Данный срипт запускает веб сервис Gradio для работы с моделью предсказания классификации ирисов

## Dockerfile
Файл для создания докерконтейнера

## Сборка docker контейнера
docker build -t lab3 . 

## Запуск docker контейнера
docker run -p 7860:7860 lab3 