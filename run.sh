#!/usr/bin/env bash

if [ $# -ne 2 ]; then
	echo $0: Usage: \<Warc-key\> \<Input-File-Path\> 
	exit 1
fi

PYSPARK_PYTHON=$(readlink -f $(which python)) ~/spark-2.1.2-bin-without-hadoop/bin/spark-submit --master yarn A1_WDPS1708.py $1 $2