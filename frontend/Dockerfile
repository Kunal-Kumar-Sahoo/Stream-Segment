FROM python:3.8-slim

RUN mkdir /frontend

COPY requirements.txt /frontend

WORKDIR /frontend

RUN pip3 install -r requirements.txt

COPY . /frontend

EXPOSE 8051

CMD [ "streamlit", "run", "app.py" ]