#!/bin/bash

echo "Setting up venv on a uni Linux system. Note: use python3.10 -m venv venv for setup on GPU."

/usr/local/python/bin/python3.10 -m venv venv
source venv/bin/activate
pip install pipenv
pipenv install
