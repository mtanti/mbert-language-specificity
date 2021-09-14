#!/bin/bash
set -e

source venv_mufins/bin/activate

echo "#########################################"
echo "mypy"
echo "..checking mufins"
python -m mypy mufins
for FNAME in `find -L bin -name "*.py"`
do
    echo "..checking $FNAME"
    python -m mypy $FNAME
done
for FNAME in `find -L tools -name "*.py"`
do
    echo "..checking $FNAME"
    python -m mypy $FNAME
done
for FNAME in `find -maxdepth 1 -name "*.py"`
do
    echo "..checking $FNAME"
    python -m mypy $FNAME
done
echo ""

echo "#########################################"
echo "pylint"
echo "..checking mufins"
python -m pylint mufins
for FNAME in `find -L bin -name "*.py"`
do
    echo "..checking $FNAME"
    python -m pylint $FNAME
done
for FNAME in `find -L tools -name "*.py"`
do
    echo "..checking $FNAME"
    python -m pylint $FNAME
done
for FNAME in `find -maxdepth 1 -name "*.py"`
do
    echo "..checking $FNAME"
    python -m pylint $FNAME
done
echo ""

echo "#########################################"
echo "project validation"
python tools/validate_project.py
echo ""

echo "#########################################"
echo "sphinx"
cd docs
make html
echo ""

echo "#########################################"
echo "unittest"
cd ../mufins/tests
python -m unittest
echo ""
