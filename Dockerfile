FROM ubuntu:latest

RUN apt update
WORKDIR /usr/app/code
COPY ./code ./
COPY ./data ../data

RUN apt install python3-pip -y
RUN pip install -r requirements.txt

CMD ./run_scripts.sh

