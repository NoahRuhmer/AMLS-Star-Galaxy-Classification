#!/bin/bash
echo
echo --------------------------------------------------------
echo --     Starting AMLS - Star/Galaxy Classification     --
echo --------------------------------------------------------
echo
python3 -u 01_alignment.py
python3 -u 02_preparation.py
python3 -u 04_augmentation.py
python3 -u 03_model.py
echo
echo --------------------------------------------------------
echo --     Finished AMLS - Star/Galaxy Classification     --
echo --------------------------------------------------------