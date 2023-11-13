FROM python:3.7

RUN wget -O /usr/local/bin/dumb-init https://github.com/Yelp/dumb-init/releases/download/v1.2.5/dumb-init_1.2.5_x86_64 && chmod +x /usr/local/bin/dumb-init

WORKDIR /code

COPY web-visu/requirements.txt /code

RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt

COPY . /code

CMD ["uwsgi", "--die-on-term", "uwsgi.ini"]
