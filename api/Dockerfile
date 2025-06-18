# Dockerfile
FROM ubuntu:22.04

# 1) system dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      curl \
      ffmpeg \
      git \
      python3-pip \
      python3-dev \
      libpq-dev \
      rustc \
      cargo \
 && rm -rf /var/lib/apt/lists/*

# 2) make sure pip, numpy & scipy are there
RUN pip3 install --upgrade pip \
 && pip3 install numpy scipy

# 3) force generic (no-dotprod) builds of llama.cpp
ENV CFLAGS="-O3 -fPIC" \
    CXXFLAGS="-O3 -fPIC"

# 4) install the rest of your Python deps
WORKDIR /app
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# 5) spaCy model (so spacy.load("en_core_web_sm") works)
RUN python3 -m spacy download en_core_web_sm

# 6) copy your code
COPY . .

# 7) create directories for persistent volumes
RUN mkdir -p /app/chroma_db /app/storage /app/outputs \
 && chmod 755 /app/chroma_db /app/storage /app/outputs

# 8) default Postgres connection settings (compose will override USER/PASSWORD/DB if you like)
ENV PGHOST=db \
    PGPORT=5432 \
    PGUSER=myuser \
    PGPASSWORD=mypassword \
    PGDATABASE=mydb

EXPOSE 5000

# 9) launch your app
CMD ["python3", "app.py"]
