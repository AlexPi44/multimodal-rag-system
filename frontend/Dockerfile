FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y nodejs npm && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir reflex

COPY . .

RUN reflex init || true

EXPOSE 3000
CMD ["reflex", "run", "--env", "prod"]
