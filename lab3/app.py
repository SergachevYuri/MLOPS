import gradio as gr
from transformers import pipeline

# Загрузка модели через pipeline
pipe = pipeline("text-classification", model="arinakosovskaia/implicit_toxicity")

def predict_toxicity(text):
    # Используем пайплайн для классификации текста
    result = pipe(text)
    # Форматируем вывод, чтобы показать только метку и вероятность
    label = result[0]['label']
    score = result[0]['score']
    return f"Label: {label}, Score: {round(score, 4)}"

# Создаем интерфейс Gradio
interface = gr.Interface(
    fn=predict_toxicity,
    inputs="text",
    outputs="text",
    title="Модель для определения токсичности текста",
    description="Эта модель используется для определения токсичности введенного текста. Введите текст ниже и нажмите 'Submit' для получения результатов."
)

# Запускаем интерфейс
interface.launch()
