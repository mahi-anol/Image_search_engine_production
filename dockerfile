FROM python:3.12.11-bookworm

WORKDIR /

COPY /src /src/
COPY /setup.py /

COPY /requirements.txt /

RUN pip install --upgrade pip 

RUN pip install --no-cache-dir -r requirements.txt

COPY /.env /


EXPOSE  7192

RUN pip install gunicorn

COPY ./data/processed/classes.json /data/processed/
COPY ./checkpoints/cross_entropy_best.pt ./checkpoints/
CMD ["uvicorn", "src.api.search_system:app","--host", "0.0.0.0", "--port", "7192"]

