# WDPS1708
Web Data Processing System Assignments - Group 8 - 2017/2018

Fatima Garcia
Gerasimos M. Giannos
Vasiliki Kalaitzi

Large Scale Entity Linking

#Basic funcionality#
This project consists of a python program to perform entity linking by processing WARC files. It can be run on a spark cluster. The main steps that are taken to do so are the following ones:
1. Extract plain text from the content of the HTML pages in the WARC file. 
2. Perform NLP and NER to identify the entities in the content
3. Link the entities to a Freebase database making an ELASTIC SEARCH query.
4. By using the IDS retrieve from the ELASTIC SEARCH query, link the entities to the "mother KB"
5. Filter the 10 best results by 'match' (log(facts)*score) to perform entity disambiguation.
6. Retrieve dbpedia links of the entities associated with the highest match and extract the text from the abstract. 
7. Compare this text with the one extracted from the WARC file by using the cos similarity. 
8. Return for each entity, the freebaseID that has a highest similarity

#Main dependencies#
--- Python packages ---



#How to
