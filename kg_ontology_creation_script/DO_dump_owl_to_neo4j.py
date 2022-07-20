# -*- coding:utf-8 -*-
#This script help to load disease ontology information into neo4j

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
		graph.schema.create_uniqueness_constraint("Disease", "url")
	except:
		pass

	node_matcher = NodeMatcher(graph)
	rel_matcher = RelationshipMatcher(graph)

	key_subset = set(("subClassOf","url","name","definition","hasAlternativeId","cross reference","hasExactSynonym","id"))
	
	num_file = sum([1 for i in open(target_file,"r")])
	target_node_dict = {}
	with open(target_file, "r") as f:
		for idx, line in tqdm(enumerate(f),total=num_file):
			line_s = line.rstrip("\n").split("\t")
			target_node = Node("Disease")
			nodel_url = line_s[-1].split("|",1)[1]
			target_node["url"] = nodel_url
			target_node["source"] = "Human Disease Ontology"
			for info in line_s[:-1]:
				node_k, node_v = info.split("|",1)
				if node_k in key_subset:
					if node_k != "subClassOf":
						if not node_k in target_node:
							target_node[node_k] = node_v
						else:
							target_node[node_k] += (","+node_v)
					else:
						target_node_dict[nodel_url] = node_v
			if "url" in target_node:
				if not node_matcher.match("Disease",url=target_node["url"]):
					tx = graph.begin()
					tx.create(target_node)
					graph.commit(tx)
	target_node_dict_len = len(target_node_dict)
	for s_node_url, o_node_url in tqdm(target_node_dict.items(),total=target_node_dict_len):
		s_node = node_matcher.match("Disease", url=s_node_url).first()
		o_node = node_matcher.match("Disease",url=o_node_url).first()
		if s_node and o_node:
			p_relationship_name = "subClassOf"
			p_relationship = Relationship(s_node, p_relationship_name, o_node, source="Human Disease Ontology")
			subgraph = s_node | p_relationship | o_node
			tx = graph.begin()
			tx.merge(subgraph, "Disease", "url")
			graph.commit(tx)
		else:
			target_err_f.write("\t".join([s_node_url, o_node_url])+"\n")
	target_err_f.close()
	