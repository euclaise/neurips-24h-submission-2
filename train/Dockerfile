FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update && apt-get install -y git python3-virtualenv wget

WORKDIR /workspace

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY train.py ./
COPY SlimTrainer/ ./SlimTrainer/
COPY ds.py ./

CMD [ "python", "ds.py"]
CMD [ "python", "train.py"]
