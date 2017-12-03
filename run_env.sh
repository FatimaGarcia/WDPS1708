#!/usr/bin/env bash

if [ $# -ne 2 ]; then
	echo $0: Usage: \<Warc-key\> \<Input-File-Path\> 
	exit 1
fi

if [ "$SPARK_HOME" = "" ]; then
	echo SPARK_HOME not set. Please create the environment variable SPARK_HOME pointing to your Spark home directory.
	exit 1
fi
# Create virtual environment and install all necessary dependencies
virtualenv venv -p python2.7
source venv/bin/activate
virtualenv --relocatable venv
pip install numpy
pip install bs4
pip install nltk
pip install scipy
pip install sklearn
pip install requests

zip -r venv.zip venv

PYSPARK_PYTHON=venv/lib/python2.7/sites-packages $SPARK_HOME/bin/spark-submit \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./VENV/venv/bin/python \
--master yarn \
--deploy-mode cluster \
--archives venv.zip#VENV \
A1_WDPS1708.py $1 $2
