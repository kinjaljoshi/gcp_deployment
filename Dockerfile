FROM python:3.8-slim

COPY . /mlops_hd_predict/app

WORKDIR /mlops_hd_predict/app

RUN pip install -r requirements.txt

CMD ["python","hd_predict.py"]