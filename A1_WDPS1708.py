#WDPS - Assignment 1 - Group 8
#Fatima Garcia
#Gerasimos M. Giannos
#Vasiliki Kalaitzi

from pyspark import SparkContext, SparkConf
import sys
import collections
import os
from subprocess import call, Popen
import re
import requests
import json
import math

from bs4 import BeautifulSoup
from bs4.element import Comment

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tag import StanfordNERTagger #NER 

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
#2. NER - StanfordNER

def NLP_NER(record): 
    #sent_text = nltk.sent_tokenize(record)
    tokenized_text = nltk.word_tokenize(record)
    #tag_text = nltk.pos_tag(tokenized_text)
   
    ner_text = nlp.tag(tokenized_text) #Option 1 - Word tokenization
    #ner_text = [nlp.tag(s.split()) for s in sent_text] #Option 2 - Sentece tokenization

    yield ner_text

#StanfordNERTagger - Files needed
classifier = 'stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz' #Path may change
jar = 'stanford-ner/stanford-ner.jar'   #Path may change
nlp = StanfordNERTagger(classifier,jar)

rdd_for_disambiguation = rdd_pairs #Keep RDD with plain text to do entity disambiguation 
rdd_ner = rdd_pairs.flatMapValues(NLP_NER) #RDD tuples (key, tuple(word, label))

#print(rdd_ner.collect())

#Extract Name Entities from result - Function to get recognized entities from Stanford NER
def get_entities_StanfordNER(record):
    entities = []
    for i in record:
        if i[1] !='O' and i[0] not in entities:
            entities.append(i[0])
    yield entities

rdd_ner_entities = rdd_ner.flatMapValues(get_entities_StanfordNER) #RDD tuples (key, entities)

#print(rdd_ner_entities.collect())

#
ELASTICSEARCH_URL = 'http://10.149.0.127:9200/freebase/label/_search'
def get_label(record):
	ids = []
	triples = {}
	for i in record:
		query = i
		response = requests.get(ELASTICSEARCH_URL, params={'q': query, 'size':100})
		if response:
		    response = response.json()
		    for hit in response.get('hits', {}).get('hits', []):
        		freebase_id = hit.get('_source', {}).get('resource')
        		label = hit.get('_source', {}).get('label')
	       		score = hit.get('_score', 0)
	       		if freebase_id not in ids:
	       			ids.append(freebase_id)
	        		triples[freebase_id]= ({'label': label, 'score': score})
        		else:
        			score_1 = max(triples.get(freebase_id, 'score'), score)
        			triples[freebase_id] = ({'label': label, 'score': score})
        		info = (i, triples)
	yield info

rdd_labels = rdd_ner_entities.flatMapValues(get_label)

#print(rdd_labels.collect())

TRIDENT_URL = 'http://10.141.0.11:8082/sparql'
prefixes = """
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX fbase: <http://rdf.freebase.com/ns/>
"""
same_as_template = prefixes + """
SELECT DISTINCT ?same WHERE {
    ?s owl:sameAs %s .
    { ?s owl:sameAs ?same .} UNION { ?same owl:sameAs ?s .}
}
"""
po_template = prefixes + """
SELECT DISTINCT * WHERE {
    %s ?p ?o.
}
"""

def get_facts(record):
	for i in record:
		for key in i[1]:
			response = requests.post(TRIDENT_URL, data={'print': False, 'query': po_template % key})
			if response:
				response = response.json()
				n = int(response.get('stats', {}).get('nresults',0))
				i[1]['facts'] = n

rdd_ids = rdd_labels.flatMapValues(get_facts)

print(rdd_ids.collect())