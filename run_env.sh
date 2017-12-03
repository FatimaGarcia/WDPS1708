#!/usr/bin/env bash

if [ $# -ne 2 ]; then
	echo $0: Usage: \<Warc-key\> \<Input-File-Path\> 
	exit 1
fi

if [ "$SPARK_HOME" = "" ]; then
	echo SPARK_HOME not set. Please create the environment variable SPARK_HOME pointing to your Spark home directory.
	exit 1
fi
# This assumes there is a python virtual environment in the "venv" directory
source venv/bin/activate
virtualenv --relocatable venv
zip -r venv.zip venv

PYSPARK_PYTHON=$(readlink -f $(which python)) $SPARK_HOME/bin/spark-submit/bin/spark-submit \
--conf spark.yarn.appMasterEnv.PYSPARK_PYTHON=./VENV/venv/bin/python \
--master yarn-cluster \
--archives venv.zip#VENV \
A1_WDPS1708.py $1 $2
