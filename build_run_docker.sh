#!/bin/bash
mkdir -p ./data
docker build -t amls_11918092 .
docker run -it amls_11918092 /bin/bash
