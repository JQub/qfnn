# JQuantumFlow
When you change the code,please update to github and pypi

## How to update to pypi:
cd JQuantumFlow_tutorial/
- change the version in the setup.cfg (if not, it will fail) 
- sh update2pypi.sh

## How to generate doc:
- pip install sphinx
- cd doc
- sphinx-apidoc -o source ../src/JQuantumFlow/
- make html
- cd build/html


