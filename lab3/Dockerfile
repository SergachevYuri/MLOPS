FROM python:3.9

# Установка зависимостей
RUN pip install pandas scikit-learn joblib gradio

# Создание и установка рабочей директории
WORKDIR /workspace

# Копирование всех необходимых файлов в рабочую директорию
COPY ./preprocess.py ./train.py ./app.py ./

# Предварительная обработка данных и обучение модели
RUN python preprocess.py
RUN python train.py

# Проверка наличия файла модели
RUN ls -l  # Это покажет, создался ли файл модели

# Запуск приложения
CMD ["python", "app.py"]
