FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

RUN apt-get update  && apt-get install -y git python3-virtualenv wget 

WORKDIR /workspace
# Setup server requriements
COPY ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt


ENV HUGGINGFACE_TOKEN="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"
ENV HUGGINGFACE_REPO="euclaise/Zen-7B_neurips"

# Copy over single file server
COPY ./main.py main.py
COPY ./api.py api.py
# Run the server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
