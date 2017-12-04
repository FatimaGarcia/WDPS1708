# WDPS1708
Web Data Processing System Assignments - Group 8 - 2017/2018

- Fatima Garcia
- Gerasimos M. Giannos
- Vasiliki Kalaitzi

# Large Scale Entity Linking

#Basic funcionality#

This project consists of a python program to perform entity linking by processing WARC files. The main steps that are taken to do so are the following ones:
1. Extract plain text from the content of the HTML pages in the WARC file. 
2. Perform NLP and NER to identify the entities in the content
3. Link the entities to a Freebase database making an ELASTIC SEARCH query.
4. By using the IDS retrieved from the ELASTIC SEARCH query, link the entities to the "mother KB"
5. Filter the 5 best results by 'match' (log(facts)*score) to perform entity disambiguation.
6. Retrieve the abstract content from dbpedia links of the entities associated with the highest match.
7. Compare this text with the one extracted from the WARC file by using the cos similarity. 
8. Return for each entity, the freebaseID that has a highest similarity.

#Main dependencies#

--- Python packages ---
The following python packages has to be installed in order to use the program: numpy , scipy, sklearn, bs4, nltk, six
requests

Also, APACHE SPARK and files from STANDFOR NER (included in this repository) are needed. 

#How to

1. Get the project (git clone https://github.com/FatimaGarcia/WDPS1708).

2. Set SPARK_HOME to your Spark home directory (e.g in the cluster ~/spark-2.1.2-bin-without-hadoop).

3. Makes sure that .sh has the correct permissions to execute it. 

4. The program takes two arguments : <WARC-KEY> <Input-file> - ./run.sh <WARC-KEY> <Input-file>

5. The output is a file (part-00000) that is saved in the directory output.tsv in the cluster hdfs. Make sure that this directory does not exist before executing the project.
This file contains WARC-Key of the WARC file to process, the name of the entities recognized and the corresponding Freebase entity ID.


#Notes
 - The program is running using a TRIDENT server launched in node091. So we are using the following TRIDENT_URL 'http://10.141.0.125:9001/sparql'. This variable is defined in line 184 of A1_WDPS1708.py, in case it has to be changed.

 - We include two scripts to run the program:
 	1 run.sh runs it in local mode of Spark
 	2 run_env.sh should run it in Yarn mode. To run this, you should have a virtual environment venv inside the project folder with python dependencies installed

- For extracting name entities we developed two different functions:
	1. get_entities_StanfordNER - Extract all the entities pointed up by standford NER as single name entities.
	2. get_entities_StanfordNER_multiterm - Extract multiterm entities by using the following rule: if two adjacent named entities in the text are marked as the same type, they belong to the same entity. 
You can choose from using one or other by changing line 150 of the main script (rdd_ner_entities = rdd_ner.flatMapValues(get_entities_StanfordNER))

- As we only managed to run it locally, we are performing the entity disambiguation with 5 results with a highest match (math.log(facts)*score). That is that we are extracting dbpedia text and calculating cosine similarity with only that five results. 
To improve the recall, you can increase the number of entries process by changing line 228 (best_matches = dict(sorted(i[1].items(), key=lambda x:(x[1]['match']), reverse=True)[:5]))
