from pyspark import SparkContext, SparkConf
import sys
import collections
import os
from subprocess import call, Popen
import re
import requests
import json
import math
import numpy
import urllib

from bs4 import BeautifulSoup
from bs4.element import Comment

import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.tag import StanfordNERTagger #NER 
from nltk.stem.porter import PorterStemmer

#nltk.download('words')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import gensim
from stop_words import get_stop_words


#Check input parameters
if len(sys.argv) < 3 or len(sys.argv) >3:
	print('Usage - <Warc_key> <Input_file>')
else:
	record_attribute = sys.argv[1]
	in_file = sys.argv[2]

sc = SparkContext.getOrCreate()

#Read warc file and split in WARC/1.0
rdd = sc.newAPIHadoopFile(in_file,
    "org.apache.hadoop.mapreduce.lib.input.TextInputFormat",
    "org.apache.hadoop.io.LongWritable",
    "org.apache.hadoop.io.Text",
    conf={"textinputformat.record.delimiter": "WARC/1.0"})


#Process WARC file -- Convert RDD to tuples (key (WARC-Record-ID), value (Text from HTML content))
#1. Get the key
#2. Get HTML content to each page (and associated it to the corresponding key)
#3. Get Text from the HTML content

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

rdd_pairs = rdd.flatMap(processWarcfile) 

def get_entities_StanfordNER(record):
    entities = []
    for i in record:
		if (i[1] !='O' and i[0] not in entities):
			entities.append(i[0])
    
    yield entities

def lda(record):
	en_stop = get_stop_words('en')
	tokenized_text = nltk.word_tokenize(record)
	tokenized_text = [x.encode('utf-8') for x in tokenized_text]
	tokenized_text = [i for i in tokenized_text if not i in en_stop]
 	#StanfordNER
 	ner_text_NER = nlp.tag(tokenized_text) #Option 1 - Word tokenization

 	yield ner_text_NER

classifier = 'stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz' #Path may change
jar = 'stanford-ner/stanford-ner.jar'   #Path may change
nlp = StanfordNERTagger(classifier,jar)

rdd_result = rdd_pairs.flatMap(lda)
rdd_result = rdd_result.flatMap(get_entities_StanfordNER)

print(rdd_result.collect())

# turn our tokenized documents into a id <-> term dictionary
dictionary = corpora.Dictionary(i for i in rdd_result.collect())
	    
# convert tokenized documents into a document-term matrix
corpus = [dictionary.doc2bow(text) for text in rdd_result.collect()]

# generate LDA model
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=2, num_words=4))