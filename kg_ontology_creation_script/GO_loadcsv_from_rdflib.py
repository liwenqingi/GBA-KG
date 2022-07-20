# -*- coding:utf-8 -*-
#extract gene ontology information from neo4j

import regex
import sys
import py2neo
import rdflib
from rdflib import Graph,OWL,RDFS
from rdflib.term import *
import pandas as pd
from tqdm import tqdm

#information transformation
def dict_settings():
	class_dict = {"OGG_0000000009":"modification date","OGG_0000000006":"entrez id","OGG_0000000017":\
	"chromosome","OGG_0000000015":"organism NCBITaxon ID","OGG_0000000008":"gene map location",\
	"IAO_0000118":"alternative term","OGG_0000000004":"gene symbol","OGG_0000000029":"GO association",\
	"OGG_0000000030":"PubMed association","OGG_0000000007":"NCBI LocusTag","IAO_0000119":"source",\
	"OGG_0000000005":"full name","OGG_0000000018":"type of gene","IAO_0000115":"definition","rdf-schema#label":\
	"name","rdf-schema#subClassOf":"subClassOf","rdf-schema#comment":"comment","description":"description",\
	"oboInOwl#hasDbXref":"cross reference","oboInOwl#hasExactSynonym":"hasExactSynonym","oboInOwl#hasRelatedSynonym":\
	"hasRelatedSynonym","oboInOwl#id":"id","oboInOwl#hasOBONamespace":"hasOBONamespace","oboInOwl#hasBroadSynonym":\
	"hasBroadSynonym","oboInOwl#hasAlternativeId":"hasAlternativeId","oboInOwl#inSubset":"inSubset","RO_0002175":\
	"present in taxon","RO_0002161":"never in taxon","UBPROP_0000008":"taxon notes","IAO_0000111":"IAO_0000111","inchi":"inchi",\
	"formula":"formula","inchikey":"inchikey"}
	return class_dict
#pattern match
def pattern_match():
	pattern = regex.compile("http://purl.obolibrary.org/obo/GO_[0-9]*")
	return pattern
#rdflib loading
def load_rdf_to_rdflib(target_file):
	g = Graph()
	res_graph = g.parse(target_file,format="application/rdf+xml")
	return res_graph

if __name__ == "__main__":
	rdf_file = sys.argv[1]
	

	class_dict = dict_settings()
	#loading 
	res_graph = load_rdf_to_rdflib(rdf_file)
	pattern = pattern_match()
	#write
	with open(rdf_file + ".biological_process.csv", "w") as O:
		for s,p,o in tqdm(res_graph.triples((None,\
			URIRef("http://www.geneontology.org/formats/oboInOwl#hasOBONamespace"),\
			Literal("biological_process")))):
			if pattern.match(s):
				for s1,p1,o1 in res_graph.triples((s,None,None)):
					p1_predication = p1.split("/")[-1]
					if p1_predication in class_dict:
						o1 = o1.replace("\n","")
						O.write("|".join([class_dict[p1_predication],o1])+"\t")
				O.write("url|"+str(s)+"\n")
	with open(rdf_file + ".cellular_component.csv", "w") as O1:
		for s,p,o in tqdm(res_graph.triples((None,\
			URIRef("http://www.geneontology.org/formats/oboInOwl#hasOBONamespace"),\
			Literal("cellular_component")))):
			if pattern.match(s):
				for s1,p1,o1 in res_graph.triples((s,None,None)):
					p1_predication = p1.split("/")[-1]
					if p1_predication in class_dict:
						o1 = o1.replace("\n","")
						O1.write("|".join([class_dict[p1_predication],o1])+"\t")
				O1.write("url|"+str(s)+"\n")
	with open(rdf_file + ".molecular_function.csv", "w") as O2:
		for s,p,o in tqdm(res_graph.triples((None,\
			URIRef("http://www.geneontology.org/formats/oboInOwl#hasOBONamespace"),\
			Literal("molecular_function")))):
			if pattern.match(s):
				for s1,p1,o1 in res_graph.triples((s,None,None)):
					p1_predication = p1.split("/")[-1]
					if p1_predication in class_dict:
						o1 = o1.replace("\n","")
						O2.write("|".join([class_dict[p1_predication],o1])+"\t")
				O2.write("url|"+str(s)+"\n")