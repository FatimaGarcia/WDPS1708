from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf

import sys
import collections
import os
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
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import string


#nltk.download('words')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import gensim
from stop_words import get_stop_words
import pyLDAvis.gensim 

#Check input parameters
if len(sys.argv) < 5 or len(sys.argv) >5:
	print('Usage - <Warc_key> <Input_file> <Mode> <Number of topics>')
else:
	record_attribute = sys.argv[1]
	in_file = sys.argv[2]
	rec_mode = sys.argv[3]
	topic_number = sys.argv[4]

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


def clean_text(record):
	en_stop = get_stop_words('en')
	punctuation = set(string.punctuation) 
	lemmatize = WordNetLemmatizer()

	tokenized_text = nltk.word_tokenize(record)
	tokenized_text = [x.encode('utf-8') for x in tokenized_text]
	tokenized_text = [i for i in tokenized_text if i not in en_stop]
	tokenized_text = [i for i in tokenized_text if i not in punctuation]
 	tokenized_text = [lemmatize.lemmatize(i) for i in tokenized_text]
 	if rec_mode == 1:
 		tokenized_text = nlp.tag(tokenized_text) #Option 1 - Word tokenization

 	yield tokenized_text

def get_entities_StanfordNER(record):
    entities = []
    for i in record:
		if (i[1] !='O' and i[0] not in entities):
			entities.append(i[0])
    
    yield entities

classifier = 'stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz' #Path may change
jar = 'stanford-ner/stanford-ner.jar'   #Path may change
nlp = StanfordNERTagger(classifier,jar)

rdd_pairs = rdd.flatMap(processWarcfile) 
rdd_result = rdd_pairs.flatMap(clean_text)
if rec_mode == 1:
	rdd_result = rdd_result.flatMap(get_entities_StanfordNER)

#Convert RDD to dataframe
df = rdd_result.map(lambda x: (x, )).toDF(schema=['text'])

#LDA
entities = df.select('text').collect()
text_list =[]
for i in entities:
	if i[0]:
		text_list.append(i[0])

dictionary = corpora.Dictionary(text_list)
corpus = [dictionary.doc2bow(j) for j in text_list]	 

ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=topic_number, id2word = dictionary, passes=20)

print(ldamodel.print_topics(num_topics=topic_number, num_words=5))

data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
pyLDAvis.save_html(data,'LDA_visualization.html')