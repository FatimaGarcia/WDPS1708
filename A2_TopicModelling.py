#WDPS - Assignment 2 - Group 8
#Fatima Garcia
#Gerasimos M. Giannos
#Vasiliki Kalaitzi

#Packages
#Pyspark
from pyspark.sql import SparkSession
from pyspark import SparkContext, SparkConf
#General packages
import sys
import collections
import os
import re
import requests
import json
import numpy
import urllib
import string
from stop_words import get_stop_words
#Newspaper articles
import newspaper
from newspaper import Article
import time
#BeatifulSOup
from bs4 import BeautifulSoup
from bs4.element import Comment
#NLTK
import nltk
from nltk.tag import StanfordNERTagger #NER 
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('words')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

#LDA - NMF
import scipy
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora, models
import gensim
import pyLDAvis.gensim 

#Input parameters
input_mode = sys.argv[1].upper()
rec_mode = sys.argv[2]
topic_number = sys.argv[3]
topic_mode = sys.argv[4]
if (input_mode == 'WARC'):
    in_file = sys.argv[5]
if (sys.argv[1] == 'help'):
    print('Usage: <Input to process - WARC or ARTICLE> <Topic modelling mode: 1 - Entities, 2-Full text> <Number of topics> <Topic modelling method - 1-LDA, 2-NMF> [If WARC: <Input_file>]')

#Spark config and session
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

#Functions
#Function to get only visible text in HTML - From https://stackoverflow.com/questions/1936466/beautifulsoup-grab-visible-webpage-text
def tag_visible(element):
    #Filter elements with following tags
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    #Filter comments
    if isinstance(element, Comment):
        return False
    return True

#Extract text from HTML
def get_text(html):
    soup = BeautifulSoup(html, "html5lib")  #Extract HTMLContent
    plain_text = soup.findAll(text=True) #Get plain text
    value = filter(tag_visible, plain_text) #Get only visible text
    #Format the text    
    value = " ".join(value) 
    value = re.sub(r'[^\x00-\x7F]+',' ', value) #Replace special unicode characters
    value = re.sub(r'[(?<=\{)(:*?)(?=\})]', ' ', value) #Replace special characters
    value = ' '.join(value.split())

    return value

#Function to extract plain text from HTML content of the WARC file
def processWarcfile(record):
    _, payload = record
    value = ''
    #Check if the WARC block contains HTML code
    if ('<html' in payload):
       html = payload.split('<html')[1]
       value = get_text(html)
    yield value

#Natural language processing of the plain text: tokenization, removing stop words and lemmatization
def NLP(record):
    en_stop = get_stop_words('en')
    punctuation = set(string.punctuation) 

    tokenized_text = nltk.word_tokenize(record)
    tokenized_text = [x.encode('utf-8') for x in tokenized_text]
    tokenized_text = [i for i in tokenized_text if i not in en_stop]

    if rec_mode == 1: #If topic modelling with entities
        tokenized_text = nlp.tag(tokenized_text) #Option 1 - Word tokenization

    yield tokenized_text

#If rec mode == 1 - Topic modelling with entities - Function to extract Named entities
def get_entities_StanfordNER(record):
    entities = []
    for i in record:
        if (i[1] !='O' and i[0] not in entities):
            entities.append(i[0])
    
    yield entities

#Option 1 - Topic modelling from CNN articles 
if input_mode == 'ARTICLE':
    actual_date = time.strftime("%Y/%m/%d")
    #Get CNN articles
    date_articles = []
    cnn_paper = newspaper.build('http://cnn.com', memoize_articles=False)
    for article in cnn_paper.articles:
        if str(actual_date) in article.url:
            article = Article(article.url, keep_article_html=True)
            article.download()
            article.parse()
            date_articles.append(get_text(article.html))
    rdd = sc.parallelize(date_articles)

#Option 2 - Topic modelling from WARC file.
if input_mode == 'WARC':
    #Read warc file and split in WARC/1.0
    rdd = sc.newAPIHadoopFile(in_file,
        "org.apache.hadoop.mapreduce.lib.input.TextInputFormat",
        "org.apache.hadoop.io.LongWritable",
        "org.apache.hadoop.io.Text",
        conf={"textinputformat.record.delimiter": "WARC/1.0"})

    rdd = rdd.flatMap(processWarcfile) 

#Common processing for both options
classifier = 'stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz' #Path may change
jar = 'stanford-ner/stanford-ner.jar'   #Path may change
nlp = StanfordNERTagger(classifier,jar)

rdd_result = rdd.flatMap(NLP)
if rec_mode == 1:
    rdd_result = rdd_result.flatMap(get_entities_StanfordNER)
    
print(rdd_result.collect())
