# -*- coding:utf-8 -*-
#This script help to load ncbitaxon bacteria information into neo4j
import sys
import py2neo
import numpy as np
import pandas as pd
from tqdm import tqdm
from py2neo import Node, Graph, NodeMatcher, Relationship,RelationshipMatcher

if __name__ == "__main__":
	target_file = sys.argv[1]
	target_err_log = target_file + ".error_log"

	target_err_f = open(target_err_log, "w")

	graph = Graph("http://localhost:7474", auth=("neo4j","123"))
	#delete constraint
	#graph.schema.drop_uniqueness_constraint("Cellular component", "id")
	try:
		graph.schema.create_uniqueness_constraint("Microbe", "url")
	except:
		pass

	node_matcher = NodeMatcher(graph)
	rel_matcher = RelationshipMatcher(graph)
	
	num_file = sum([1 for i in open(target_file,"r")])
	target_node_dict = {}
	with open(target_file, "r") as f:
		for idx, line in tqdm(enumerate(f),total=num_file):
			line_s = line.rstrip().split("\t")
			target_node = Node("Microbe")
			#nodel_url = line_s[2].split("|",1)[1]
			#target_node["url"] = nodel_url
			nodel_url = ""
			node_subclassOf = ""
			for info in line_s:
				node_k, node_v = info.split("|",1)
				if node_k != "subclassOf":
					if node_k == "url":
						nodel_url = node_v
					if not node_k in target_node:
						target_node[node_k] = node_v
					else:
						target_node[node_k] += (","+node_v)
				else:
					node_subclassOf = node_v
			if nodel_url and node_subclassOf:
				target_node_dict[nodel_url] = node_subclassOf
			target_node["source"] = "NCBI organismal classification"
			if "url" in target_node:
				if not node_matcher.match("Microbe",url=target_node["url"]):
					tx = graph.begin()
					tx.create(target_node)
					graph.commit(tx)
	target_node_dict_len = len(target_node_dict)
	for s_node_url, o_node_url in tqdm(target_node_dict.items(),total=target_node_dict_len):
		s_node = node_matcher.match("Microbe", url=s_node_url).first()
		o_node = node_matcher.match("Microbe",url=o_node_url).first()
		if s_node and o_node:
			p_relationship_name = "subClassOf"
			p_relationship = Relationship(s_node, p_relationship_name, o_node, source="ncbitaxon")
			subgraph = s_node | p_relationship | o_node
			tx = graph.begin()
			tx.merge(subgraph, "Microbe", "url")
			graph.commit(tx)
		else:
			target_err_f.write("\t".join([s_node_url, o_node_url])+"\n")
	target_err_f.close()
	