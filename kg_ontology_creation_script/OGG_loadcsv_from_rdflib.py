# -*- coding:utf-8 -*-
#extract OGG ontology information

import regex
import sys
import py2neo
import rdflib
from rdflib import Graph,OWL,RDFS
from rdflib.term import URIRef
import pandas as pd
from tqdm import tqdm
#usage class settings
def ogg_dict_setttings():
	class_dict = {"OGG_0000000009":"modification date","OGG_0000000006":"entrez id","OGG_0000000017":\
	"chromosome","OGG_0000000015":"organism NCBITaxon ID","OGG_0000000008":"gene map location",\
	"IAO_0000118":"alternative term","OGG_0000000004":"gene symbol","OGG_0000000029":"GO association",\
	"OGG_0000000030":"PubMed association","OGG_0000000007":"NCBI LocusTag","IAO_0000119":"source",\
	"OGG_0000000005":"full name","OGG_0000000018":"type of gene","IAO_0000115":"definition","rdf-schema#label":\
	"name","rdf-schema#subClassOf":"subClassOf","rdf-schema#comment":"comment","description":"description",\
	"oboInOwl#hasDbXref":"cross reference","oboInOwl#hasExactSynonym":"hasExactSynonym","oboInOwl#hasRelatedSynonym":\
	"hasRelatedSynonym","oboInOwl#id":"id","oboInOwl#hasOBONamespace":"hasOBONamespace","oboInOwl#hasBroadSynonym":\
	"hasBroadSynonym","oboInOwl#hasAlternativeId":"hasAlternativeId","oboInOwl#inSubset":"inSubset","RO_0002175":\
	"present in taxon","RO_0002161":"never in taxon","UBPROP_0000008":"taxon notes","IAO_0000111":"IAO_0000111"}
	return class_dict
#pattern
def ogg_pattern_match():
	pattern = regex.compile("http://purl.obolibrary.org/obo/OGG_[0-9]*")
	return pattern

def load_rdf_to_rdflib(target_file):
	g = Graph()
	res_graph = g.parse(target_file,format="application/rdf+xml")
	return res_graph

if __name__ == "__main__":
	rdf_file = sys.argv[1]
	output_csv = rdf_file + ".csv"

	ogg_class_dict = ogg_dict_setttings()

	res_graph = load_rdf_to_rdflib(rdf_file)
	pattern = ogg_pattern_match()
	#write
	with open(output_csv, "w") as O:
		for s,p,o in tqdm(res_graph.triples((None,None,OWL.Class))):
			if pattern.match(s):
				for s1,p1,o1 in res_graph.triples((s,None,None)):
					p1_predication = p1.split("/")[-1]
					if p1_predication in ogg_class_dict:
						O.write("|".join([ogg_class_dict[p1_predication],o1])+"\t")
				for s2,p2,o2 in res_graph.triples((s,RDFS.subClassOf,None)):
					if pattern.match(o2):
						O.write("subClassOf|"+str(o2)+"\t")
				O.write("url|"+str(s)+"\n")
	