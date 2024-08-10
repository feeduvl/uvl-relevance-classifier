FROM python:3.11-slim-buster

RUN apt-get update && apt-get install -y \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --upgrade pip

WORKDIR /app
COPY . /app/

RUN pip3 install --no-cache-dir .

RUN mkdir -p /usr/share/nltk_data
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt


ARG mlflow_tracking_password
ARG mlflow_tracking_username
ARG mlflow_tracking_uri

ENV MLFLOW_TRACKING_USERNAME=$mlflow_tracking_username
ENV MLFLOW_TRACKING_PASSWORD=$mlflow_tracking_password
ENV MLFLOW_TRACKING_URI=$mlflow_tracking_uri

ENV TRANSFORMERS_CACHE=/app/temp/transformers/
RUN mkdir -p $TRANSFORMERS_CACHE

RUN jupyter nbconvert --to python --execute ComponentRelevanceClassifierServiceSetup.ipynb

RUN chmod +x start.sh

EXPOSE 9698

CMD ["./start.sh"]
