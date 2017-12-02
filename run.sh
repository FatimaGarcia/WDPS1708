#!/usr/bin/env bash

if [ $# -ne 2 ]; then
	echo $0: Usage: \<Warc-key\> \<Input-File-Path\> 
	exit 1
fi

if [ "$SPARK_HOME" = "" ]; then
	echo SPARK_HOME not set. Please create the environment variable SPARK_HOME pointing to your Spark home directory.
	exit 1
fi

PYSPARK_PYTHON=$(readlink -f $(which python)) $SPARK_HOME/bin/spark-submit --master yarn A1_WDPS1708.py $1 $2