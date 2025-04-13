#!/bin/bash

conda env create -f setup/env.yml

conda activate flow && pip install -r setup/requirements.txt