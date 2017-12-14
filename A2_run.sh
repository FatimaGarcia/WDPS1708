#!/usr/bin/env bash

if [ $# -ne 4 ]; then
	echo $0: Usage: \<Mode - WARC or ARTICLE\> \<If ARTICLE - News Date (Y/M/D), If WARC - Input-File-Path\> \<1 (Entities) - 2 (Full text)\> \<Number of topics\> 
	exit 1
fi

if [ "$SPARK_HOME" = "" ]; then
	echo SPARK_HOME not set. Please create the environment variable SPARK_HOME pointing to your Spark home directory.
	exit 1
fi

PYSPARK_PYTHON=$(readlink -f $(which python)) $SPARK_HOME/bin/spark-submit --master local[*] A1_WDPS1708.py $1 $2 $3 $4