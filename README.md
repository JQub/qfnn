# qfnn

The "qfnn" (or QuantumFlow Neural Network Library) 
is a library to support the implementation 
of [QuantumFlow](https://www.nature.com/articles/s41467-020-20729-5)
and its extensions, including [QF-Mixer](https://arxiv.org/pdf/2109.03806.pdf) and 
[QF-RobustNN](https://arxiv.org/pdf/2109.03430.pdf).

## How to update to pypi:
cd JQuantumFlow_tutorial/
- change the version in the setup.cfg (if not, it will fail) 
- sh update2pypi.sh

## How to generate doc:
- pip install sphinx
- cd doc
- sphinx-apidoc -o source ../src/qfnn/
- make html
- cd build/html


