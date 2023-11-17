FROM python:3.11
LABEL authors="celio bueri"

WORKDIR /home/user

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "training.py"]