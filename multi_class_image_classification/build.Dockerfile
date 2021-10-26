FROM pytorch/pytorch:1.10.0-cuda11.3-cudnn8-runtime
WORKDIR /usr/src/app
COPY ./requirements.txt ./
RUN pip install -r requirements.txt
COPY . .
ENTRYPOINT [ "./run.sh" ]
