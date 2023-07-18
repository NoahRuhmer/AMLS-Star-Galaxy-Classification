# AMLS: Star/Galaxy Classification

## Overview
This project contains a full machine learning pipeline to classify stars and galaxies from the Sloan Digital Sky Survey (SDSS).
Each file solves a subtask of the assignment description.
- Data Acquisition (`code/00_download.py`)
- Frame Alignment (`code/01_alignment.py`)
- Data Preparation (`code/02_preparation.py`)
- Model Training and Evaluation (`code/03_model.py`)
- Data Augmentation (`code/04_augmentation.py`)

A detailed project description can be found in the report `report.pdf`.

## Results
The goal of this model is to classify a given coordinate into either star or galaxy.

The Model achieves an accuracy of 90.5% and a matthews correlation coefficient of 0.746 on the test frame.

Frames from rerun 301, run 8162 and camcol 6 (online view [here](https://data.sdss.org/sas/dr17/eboss/photoObj/frames/301/8162/frames-run-008162.html)) without data augmentation have been used as ML data.
Data split:
- Training fields = `[103, 111, 147, 174, 177, 214, 222]`
- Validation fields = `[120, 228]`
- Test field = `[80]`

## Setup
To run the full pipeline execute the bash script `./code/run_scripts.sh`.
This project has been tested and is working on python (3.10) with all packages from `code/requirements.txt` installed.

### Pre-Loading the Data 
This script will download ~250MB of the already stressed SDSS server. 
Please avoid this additional load on the servers and instead download the data from [this link](https://cloud.tugraz.at/index.php/s/ffwtGyCfJXEoaRp) instead.
Download the full `data` folder and place it in the projects root.
Then start the pipeline using the downloaded data with `run_scripts_no_download.sh`.

### Docker
Alternatively you may use the provided Dockerfile to automatically create a container with all required dependencies. 
To build and start the container run `build_run_docker.sh` which can take quite a while to finish.
When the docker image has been started you can use `run_scripts.sh` or preferably `run_scripts_no_download.sh` if you already downloaded the data.

## Runtime/Resources
Disk Storage Requirement: 10GB
RAM Peak usage: ~10GB

Runtime estimates (Desktop PC without GPU):
- Building Docker: ~5 minutes
- Data Download: ~30 minutes 
- Frame Alignment: ~30 minutes
- Data Preparation: ~1 minute
- Data Augmentation: ~1 minute
- Model Training: ~15 minutes