#!/bin/bash

echo "Setting up venv on a uni Linux system."

/usr/local/python/bin/python3.10 -m venv venv
source venv/bin/activate
pip install pipenv
pipenv install
