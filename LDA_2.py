import re # needed to remove special character
from pyspark import Row
import sys
from pyspark import SparkContext, SparkConf
import sys
import collections
import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import Tokenizer, CountVectorizer
from pyspark.mllib.clustering import LDA
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, LongType

import re
import requests
import json
import math
import numpy
import urllib

from bs4 import BeautifulSoup
from bs4.element import Comment

import nltk
from nltk.tag import StanfordNERTagger #NER 
#Check input parameters
if len(sys.argv) < 3 or len(sys.argv) >3:
	print('Usage - <Warc_key> <Input_file>')
else:
	record_attribute = sys.argv[1]
	in_file = sys.argv[2]


sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

#Read warc file and split in WARC/1.0
rdd = sc.newAPIHadoopFile(in_file,
    "org.apache.hadoop.mapreduce.lib.input.TextInputFormat",
    "org.apache.hadoop.io.LongWritable",
    "org.apache.hadoop.io.Text",
    conf={"textinputformat.record.delimiter": "WARC/1.0"})

#Function to get only visible text in HTML - From https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text
def tag_visible(element):
    #Filter elements with following tags
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    #Filter comments
    if isinstance(element, Comment):
        return False
    return True

def get_text(html, flag):
	soup = BeautifulSoup(html, "html5lib")  #Extract HTMLContent
	if flag == 1:
		value = soup.find("span", {"property" : "dbo:abstract", "xml:lang":"en"})
		if value is not None:
			value = value.getText()
		else:
			value = ''
	else:
		plain_text = soup.findAll(text=True) #Get plain text
		value = filter(tag_visible, plain_text) #Get only visible text
		#Format the text	
		value = " ".join(value) 
		value = re.sub(r'[^\x00-\x7F]+',' ', value) #Replace special unicode characters
		value = re.sub(r'[(?<=\{)(:*?)(?=\})]', ' ', value) #Replace special characters
		value = ' '.join(value.split())

	return value

#Function to extract the WARC key and plain text from HTML content of the WARC file
def processWarcfile(record):
    _, payload = record
    value = ''
    #Check if the WARC block contains HTML code
    if ('<html' in payload):
        html = payload.split('<html')[1]
        value = get_text(html, 0)
    yield value

rdd_text = rdd.flatMap(processWarcfile) 

#print(rdd_text.collect())
df = rdd_text.map(lambda x: (x, )).toDF(schema=['text'])

row_with_index = Row(*["id"] + df.columns)

def make_row(columns):
    def _make_row(row, uid):
        row_dict = row.asDict()
        return row_with_index(*[uid] + [row_dict.get(c) for c in columns])

    return _make_row

f = make_row(df.columns)

indexed = (df.rdd
           .zipWithUniqueId()
           .map(lambda x: f(*x))
           .toDF(StructType([StructField("id", LongType(), False)] + df.schema.fields)))
# tokenize
tokenizer = Tokenizer(inputCol="text", outputCol="tokens")
tokenized = tokenizer.transform(indexed)

# remove stop words
remover = StopWordsRemover(inputCol="tokens", outputCol="words")
cleaned = remover.transform(tokenized)

# vectorize
cv = CountVectorizer(inputCol="words", outputCol="vectors")
count_vectorizer_model = cv.fit(cleaned)
result = count_vectorizer_model.transform(cleaned)


print(result.show())
