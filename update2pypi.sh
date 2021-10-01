VERSION=`cat setup.cfg  | grep '^version =' | sed 's/version = 0.1.//'`
NEW_VERSION=`expr $VERSION + 1`
sed -i "s/version = 0.1.$VERSION/version = 0.1.$NEW_VERSION/g" setup.cfg

python -m build
twine upload dist/*
echo "Wait for updating qfnn"

sleep 5
echo "Start to update qfnn"
pip install qfnn -U

pip install qfnn -U

echo "Wait for updating API document"
sleep 2
cd doc
make clean
make html
cd ..
git add --all
git commit -m "auto update"
git push
