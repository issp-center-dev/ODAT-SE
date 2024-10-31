#!/bin/sh

if [ "_$1" == "_--latex" ]; then
  BUILD_LATEX=ON
  latexmk --version > /dev/null 2>&1 || { echo "Latexmk is not installed"; exit 1; }
else
  BUILD_LATEX=OFF
fi

sphinx-build --version > /dev/null 2>&1 || { echo "Sphinx is not installed"; exit 1; }

DOCROOT=`pwd`

# user's manual
cd ${DOCROOT}/ja
make html
if [ $BUILD_LATEX = "ON" ]; then
  make latexpdf
fi

cd ${DOCROOT}/en
make html
if [ $BUILD_LATEX = "ON" ]; then
  make latexpdf
fi

export SPHINX_APIDOC_OPTIONS=members,show-inheritance

# api
cd $DOCROOT
sphinx-apidoc -M -P -f -e -d 1 -o ./api/source ../src/odatse/
cd api
make html
