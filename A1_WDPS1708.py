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
from nltk.tree import Tree

#nltk.download('words')
#nltk.download('punkt')
#nltk.download('averaged_perceptron_tagger')

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
        value = re.sub(r'[^\x00-\x7F]+',' ', value) #Replace special unicode characters
        value = re.sub(r'[(?<=\{)(:*?)(?=\})]', ' ', value) #Replace special characters
        value = ' '.join(value.split())
        yield (key, value)

rdd_pairs = rdd.flatMap(processWarcfile) #RDD with tuples (key, text)

print(rdd_pairs.collect())


#NLP - NER  
#1. Tokenization
#2. NER - StanfordNER

def NLP_NER(record): 
    #sent_text = nltk.sent_tokenize(record)
    tokenized_text = nltk.word_tokenize(record)
    tag_text = nltk.pos_tag(tokenized_text)
    
    #StanfordNER
    ner_text_NER = nlp.tag(tokenized_text) #Option 1 - Word tokenization
    #ner_text = [nlp.tag(s.split()) for s in sent_text] #Option 2 - Sentece tokenization
    
    yield ner_text_NER

#StanfordNERTagger - Files needed
classifier = 'stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz' #Path may change
jar = 'stanford-ner/stanford-ner.jar'   #Path may change
nlp = StanfordNERTagger(classifier,jar)

rdd_ner = rdd_pairs.flatMapValues(NLP_NER) #RDD tuples (key, tuple(word, label))

#print(rdd_ner.collect())

#Entity extraction
#Extract Name Entities from result - Function to get recognized entities from Stanford NER
def get_entities_StanfordNER(record):
    entities = []
    for i in record:
        if i[1] !='O' and i[0] not in entities:
            entities.append(i[0])
    yield entities

rdd_ner_entities = rdd_ner.flatMapValues(get_entities_StanfordNER) #RDD tuples (key, entities)

#print(rdd_ner_entities.collect())

#Link entities to KB
ELASTICSEARCH_URL = 'http://10.149.0.127:9200/freebase/label/_search'

#Get IDs, label and score from ELASTICSEARCH for each entity
def get_elasticsearch(record):
	tuples = []
	for i in record:
		query = i
		response = requests.get(ELASTICSEARCH_URL, params={'q': query, 'size':100}) #Query all the entities
		result = {}
		if response:
		    response = response.json()
		    for hit in response.get('hits', {}).get('hits', []):
		        freebase_id = hit.get('_source', {}).get('resource')
		        label = hit.get('_source', {}).get('label')
		        score = hit.get('_score', 0)

		        if result.get(freebase_id) == None: #Check duplicate id
		        	result[freebase_id] = ({'label':label, 'score':score, 'facts': 0 , 'match':0}) #If freebase_id is not in the dict, add all the extract info from JSON
		        else:
		        	score_1 = max(result[freebase_id]['score'], score)
		        	result[freebase_id]['score'] = score_1 #If freebase_id is in the dict, update the score but not create a new entry
		tuples.append([i, result]) #Return entity with its associated dictionary with the info from elastic search query
	yield tuples

rdd_labels = rdd_ner_entities.flatMapValues(get_elasticsearch) #RDD (key, [entity, dict{freebase_id: {score,  label}}])

#print(rdd_labels.collect())


#Link IDs to motherKB
TRIDENT_URL = 'http://10.141.0.125:9001/sparql' #May change
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
#Get number of facts from KB for each entity. Calculate the 'match' from score and facts as stated in https://github.com/bennokr/wdps2017/query.py
def get_motherKB(record):
	tuples = []
	for i in record:
		entity = i[0]
		for key in i[1]:
			response = requests.post(TRIDENT_URL, data={'print': False, 'query': po_template % key})
 			if response:
 				response = response.json()
 				n = int(response.get('stats', {}).get('nresults',0))
 				i[1][key]['facts']= n
 				i[1][key]['match'] = math.log(n) * i[1][key]['score']			    	
		tuples.append((entity, i[1]))
	yield tuples


rdd_ids = rdd_labels.flatMapValues(get_motherKB)

#print(rdd_ids.collect())

#Get 10 best matches from the complete set of results to perform entity disambiguation with the 10 best results
def get_bestmatches(record):
	best_matches = {}
	tuples = []
	for i in record:
		entity = i[0]
		best_matches = sorted(i[1].items(), key=lambda x:(x[1]['match']), reverse=True)[:1]

		#Simple version returning the result with highest match
		for key in best_matches:
			yield (entity, key)

rdd_ids_best = rdd_ids.flatMapValues(get_bestmatches)

print(rdd_ids_best.collect())

def get_ouput(record):
	with open("output.tsv", "w") as record_file:
    	record_file.write("WARC-Key\tEntity\tID\n")
    	for x in record[1]:
        	record_file.write(str(record[0])+'\t'+str(x[0])+'\t'+str(x[1])+"\n")

get_ouput(rdd_ids_best.collect())
print('The output is the file output.tsv')
