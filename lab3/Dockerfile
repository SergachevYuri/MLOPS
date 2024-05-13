FROM python:3.9

ARG GRADIO_SERVER_PORT=7860
ENV GRADIO_SERVER_PORT=${GRADIO_SERVER_PORT}

WORKDIR /app

COPY requirements.txt /app/
RUN python -m pip install --upgrade pip
RUN pip install -r /app/requirements.txt

COPY app.py /app/

ENV HOME=/workspace

CMD ["python", "/app/app.py"]