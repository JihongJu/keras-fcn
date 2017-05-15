#!/usr/bin/env bash
cd ${HOME}/workspace/
source venv/bin/activate
pip install -r requirements.txt
python setup.py build
py.test tests --cov=fcn --cov-report term-missing
