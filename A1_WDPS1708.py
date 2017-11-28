#WDPS - Assignment 1 - Group 8
#Fatima Garcia
#Gerasimos M. Giannos
#Vasiliki Kalaitzi

from pyspark import SparkContext, SparkConf
import sys
import collections
import os
from subprocess import call, Popen
import numpy
import re
import requests
import json
import math

from bs4 import BeautifulSoup
from bs4.element import Comment

import nltk
from nltk.stem import WordNetLemmatizer
#from nltk.tag.stanford import CoreNLPNERTagger #NER Option 1
from nltk.tag import StanfordNERTagger #NER Option 2
from nltk.tree import Tree

#nltk.download('words')
#nltk.download('maxent_ne_chunker')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')
#nltk.download('wordnet')
#nltk.download()  #All packages

sc = SparkContext.getOrCreate()

#Check input parameters
if len(sys.argv) < 3 or len(sys.argv) >3:
	print('Usage - <Warc_key> <Input_file>')
else:
	record_attribute = sys.argv[1]
	in_file = sys.argv[2]


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

#Function to extract the WARC key and plain text from HTML content of the WARC file
def processWarcfile(record):
    _, payload = record
    key = None
    value = None
    for line in payload.splitlines():
        if line.startswith(record_attribute):
            key = line.split(': ')[1] #Save <WARC-Key>
            break
    #Check if the WARC block contains HTML code
    if key and ('<html' in payload):
        html = payload.split('<html')[1]
        soup = BeautifulSoup(html, "html.parser")  #Extract HTMLContent
        plain_text = soup.findAll(text=True) #Get plain text
        value = filter(tag_visible, plain_text) #Get only visible text
        #Format the text
        value = " ".join(value) 
        value = re.sub(r'[^\x00-\x7F]+','', value)
        value = ' '.join(value.split())
        yield (key, value)

rdd_pairs = rdd.flatMap(processWarcfile) #RDD with tuples (key, text)

#print(rdd_pairs.collect())


#NLP - NER  
#1. Tokenization
#2. Lemmatization - Not sure if needed
#2. Tag - POS - Not sure if needed
#3. NER - StanfordNER/NLTK

def NLP_NER(record): 
    #sent_text = nltk.sent_tokenize(record)
    tokenized_text = nltk.word_tokenize(record)
    #wordnet_lemmatizer = WordNetLemmatizer()
    #lemmatize_text = [wordnet_lemmatizer.lemmatize(x) for x in tokenized_text]
    tag_text = nltk.pos_tag(tokenized_text)
   
    #Option 1 / Option 2
    ner_text = nlp.tag(tokenized_text) #Option 1 - Word tokenization
    #ner_text = [nlp.tag(s.split()) for s in sent_text] #Option 2 - Sentece tokenization

    #Option 3 - NLTK Chunks 
    #ner_text = nltk.ne_chunk(tag_text)

    yield ner_text

#Option 1 - CoreNLPNERTagger -Needed to run an external server and files needed
#call(["java", "-mx4g", "-cp", "../stanford-corenlp/*", "edu.stanford.nlp.pipeline.StanfordCoreNLPServer", "-port", "9000", "&"], stdout=None)
#nlp = CoreNLPNERTagger(url='http://localhost:9000')

#Option 2 - StanfordNERTagger - Files needed
classifier = 'stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz' #Path may change
jar = 'stanford-ner/stanford-ner.jar'   #Path may change
nlp = StanfordNERTagger(classifier,jar)

rdd_ner = rdd_pairs.flatMapValues(NLP_NER) #RDD tuples (key, tuple(word, label))

#print(rdd_ner.collect())

#Extract Name Entities from result 
#Option 1, 2 - Function to get recognized entities from Stanford NER
def get_entities_StanfordNER(record):
    entities = []
    for i in record:
        if i[1] !='O':
            entities.append(i[0])
    yield  yield entities

#Option 3 - Function to get entities from ne_chunk result - https://stackoverflow.com/questions/31836058/nltk-named-entity-recognition-to-a-python-list
def get_entities_NLTK(record):
    prev = None
    continuous_chunk = []
    current_chunk = []
    for i in record:
        if type(i) == Tree:
            current_chunk.append(" ".join([token for token, pos in i.leaves()]))
        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
            else:
                continue
    yield continuous_chunk


#rdd_ner_entities = rdd_ner.flatMapValues(get_entities_NLTK) #RDD tuples (key, entities)

rdd_ner_entities = rdd_ner.flatMapValues(get_entities_StanfordNER) #RDD tuples (key, entities)

#print(rdd_ner_entities.collect())

def get_label(record):

	ELASTICSEARCH_URL = 'http://10.149.0.127:9200/freebase/label/_search'

	for i in record:
		query = i
		response = requests.get(ELASTICSEARCH_URL, params={'q': query, 'size':100})
		ids = []
		labels = []
		scores = []
		if response:
		    response = response.json()
		    for hit in response.get('hits', {}).get('hits', []):
        		freebase_id = hit.get('_source', {}).get('resource')
        		ids.append(freebase_id)
        		label = hit.get('_source', {}).get('label')
        		labels.append(label)
        		score = hit.get('_score', 0)
        		scores.append(score)
	yield zip(ids, labels, score)

rdd_labels = rdd_ner_entities.flatMapValues(get_label)

print(rdd_labels.collect())