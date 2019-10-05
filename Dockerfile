FROM python:3

RUN rm -rf /usr/src/app
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app
USER root

RUN pip install -U setuptools
RUN pip install -U wheel
RUN pip install -U flask-cors

COPY ./requirements.txt ./
RUN pip install -r requirements.txt
COPY . /usr/src/app
RUN pip install tensorflow-gpu

CMD ["python", "-m", "run"]
CMD tail -f /dev/null
