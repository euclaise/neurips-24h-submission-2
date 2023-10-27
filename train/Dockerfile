FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update  && apt-get install -y git python3-virtualenv wget

RUN pip install -U --no-cache-dir git+https://github.com/facebookresearch/llama-recipes.git@eafea7b366bde9dc3f0b66a4cb0a8788f560c793

WORKDIR /workspace

COPY train.py ./
COPY SlimTrainer ./
COPY ds.py ./

CMD [ "python", "ds.py"]
CMD [ "python", "train.py"]

