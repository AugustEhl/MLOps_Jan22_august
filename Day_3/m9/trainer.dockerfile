# Base image
FROM python:3.7-slim

# install python
RUN apt update && \
apt install --no-install-recommends -y build-essential gcc && \
apt clean && rm -rf /var/lib/apt/lists/*

COPY ../../Day_2/m6_cc_exe/requirements.txt requirements.txt
COPY ../../Day_2/m6_cc_exe/setup.py setup.py
COPY ../../Day_2/m6_cc_exe/src/ src/
COPY ../../Day_2/m6_cc_exe/data/ data/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]
