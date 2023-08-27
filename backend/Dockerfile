FROM tiangolo/uvicorn-gunicorn:python3.8

RUN mkdir /backend

COPY requirements.txt /backend

WORKDIR /backend

RUN pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

COPY . /backend

EXPOSE 8000

CMD [ "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000" ]