python -m build
twine upload dist/*
pip install qfnn -U
cd doc
make clean
make html
cd ..
git add --all
git commit -m "auto update"
git push
