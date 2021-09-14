#!/bin/bash
set -e

python3 -m venv venv_mufins
source venv_mufins/bin/activate

pip install --upgrade pip
pip install wheel
pip install -r requirements.txt
pip install -e .

bash check_all.sh
