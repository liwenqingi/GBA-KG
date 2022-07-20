# -*- coding:utf-8 -*-
#loading compound from hetionet into knowledge graph

import sys
import pandas as pd
from tqdm import tqdm
from py2neo import Node, Graph, NodeMatcher, Relationship,RelationshipMatcher,Subgraph
'''
metaedge_dict = {"AeG":{"name":"Anatomy expresses gene","abbr":"expresses"},"Gr>G":{"name":"Gene regulates gene","abbr":"regulates"},\
"GiG":{"name":"Gene interacts gene","abbr":"interacts"},"AdG":{"name":"Anatomy downregulates gene","abbr":"downregulates"},\
"AuG":{"name":"Anatomy upregulates gene","abbr":"upregulates"},"GcG":{"name":"Gene covaries gene","abbr":"covaries"},\
"DaG":{"name":"Disease associates gene","abbr":"associates"},"DuG":{"name":"Disease upregulates gene","abbr":"upregulates"},\
"DdG":{"name":"Disease downregulates gene","abbr":"downregulates"},"DlA":{"name":"Disease localizes anatomy","abbr":"localizes"},\
"DrD":{"name":"Disease resembles disease","abbr":"resembles"},"CbG":{"name":"Compound binds gene","abbr":"binds"},"CdG":\
{"name":"Compound downregulates gene","abbr":"downregulates"},"CpD":{"name":"Compound palliates disease","abbr":"palliates"},\
"CrC":{"name":"Compound resembles compound","abbr":"resembles"},"CtD":{"name":"Compound treats disease","abbr":"treats"},\
"CuG":{"name":"Compound upregulates gene","abbr":"upregulates"}}
'''
metaedge_dict = {"CbG":{"name":"Compound binds gene","abbr":"binds"},"CdG":\
{"name":"Compound downregulates gene","abbr":"downregulates"},"CpD":{"name":"Compound palliates disease","abbr":"palliates"},\
"CrC":{"name":"Compound resembles compound","abbr":"resembles"},"CtD":{"name":"Compound treats disease","abbr":"treats"},\
"CuG":{"name":"Compound upregulates gene","abbr":"upregulates"}}
drugbank_file="/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/2_demo_owl_neo4j_3.5/2_load_data_to_neo4j_scripts/drugbank.tsv.cleaned"
metaboliteIDmapping_file=\
"/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/2_demo_owl_neo4j_3.5/2_load_data_to_neo4j_scripts/metaboliteIDmapping.txt"

global graph, match
graph_url="http://172.16.100.105:7474"
username="neo4j"
password="123"
graph = Graph(graph_url,auth=(username,password))
matcher = NodeMatcher(graph)

graph.schema.create_uniqueness_constraint("Compound","id")


def load_nodes_from_tsv(drugbank_file=drugbank_file,metaboliteIDmapping_file=metaboliteIDmapping_file):
	drugbank_dict = {}
	metaboliteIDmapping_list = []
	metaboliteIDmapping_dict = {}
	with open(metaboliteIDmapping_file,"r") as f:
		for line in f.readlines()[1:]:
			lines = line.strip().split("\t")
			if len(lines)!=1:
				chebi_id,hmdb_id,drugbank_id,name=lines[-4:]
				chebi_id,hmdb_id,drugbank_id = chebi_id.replace("\"",""),hmdb_id.replace("\"",""),\
				drugbank_id.replace("\"","")
				metaboliteIDmapping_list.append([chebi_id,hmdb_id,drugbank_id,name])
			else:
				metaboliteIDmapping_list[-1][-1] += lines[0]
		for each in metaboliteIDmapping_list:
			chebi_id,hmdb_id,drugbank_id,name=each
			if drugbank_id != "NA":
				metaboliteIDmapping_dict[drugbank_id] = {"hmdb":hmdb_id,"name":name,"chebi":chebi_id}
	with open(drugbank_file,"r") as f:
		for line in f.readlines()[1:]:
			d_id,d_name,d_type,d_groups,d_atc_codes,d_categories,d_inchikey,d_inchi,d_description=\
			line.rstrip("\n").split("\t")
			db_node = matcher.match("Compound",id=d_id)
			if not db_node:
				hmdb_id,chebi_id="",""
				if d_id in metaboliteIDmapping_dict:
					hmdb_id,chebi_id=metaboliteIDmapping_dict[d_id]["hmdb"],metaboliteIDmapping_dict[d_id]["chebi"]
				db_node = Node("Compound",id=d_id,hmdb_id=hmdb_id,chebi_id=chebi_id,\
					name=d_name,atc_codes=d_atc_codes,inchikey=d_inchikey,inchi=d_inchi,description=d_description,source="drugbank")
				tx = graph.begin()
				tx.merge(db_node,"Compound","id")
				graph.commit(tx)

def load_nodes(node_file):
	node_dict = {}
	#node_kind_set = set()
	print("loading nodes...")
	num_file = sum([1 for i in open(node_file,"r")])-1
	with open(node_file, "r") as f:
		for line in tqdm(f.readlines()[1:],total=num_file):
			identifier, name, kind = line.strip("\n").split("\t")
			identifier_s = identifier.split(":")
			kind_c, source, id_c = "", "", ""
			if len(identifier_s) == 4:
				kind_c, source, id_c = identifier_s[0], identifier_s[2], identifier_s[3]
			else:
				kind_c, id_c = identifier_s[0], identifier_s[-1]
			if not identifier in node_dict:
				node_dict[identifier] = {}
				node_dict[identifier]["source"] = source
				if kind_c == "Gene":
					node_dict[identifier]["id"] = id_c
				elif source:
					node_dict[identifier]["id"] = ":".join([source,id_c])
				else:
					node_dict[identifier]["id"] = id_c
				node_dict[identifier]["name"] = name
				node_dict[identifier]["kind"] = kind
	return node_dict

def load_edges(graph,matcher,node_file,edge_file):
	def create_subgraph(graph,matcher,metaedge):
		edge_node_name = metaedge_dict[metaedge]["name"]
		father_edge_node = matcher.match("Meta Class Relationship",name=edge_node_name)
		if not father_edge_node:
			father_edge_node = Node("Meta Class Relationship",name=edge_node_name,source="Protege")
			tx = graph.begin()
			tx.merge(father_edge_node,"Meta Class Relationship","name")
			graph.commit(tx)
		else:
			father_edge_node = father_edge_node.first()
		edge_node_rel = metaedge_dict[metaedge]["abbr"]
		child_edge_node_name = " ".join([s_node_name,edge_node_rel,t_node_name])
		child_edge_node = matcher.match("Meta Class Relationship",name=child_edge_node_name)
		child_edge_node = child_edge_node.first() if child_edge_node else Node("Meta Class Relationship",\
			name=child_edge_node_name,source="hetionet")
		edge_l_name = "_".join([edge_node_rel.upper(),"L",metaedge])
		edge_r_name = "_".join([edge_node_rel.upper(),"R",metaedge])
		l_relationship = Relationship(s_node,edge_l_name,child_edge_node,source="hetionet")
		r_relationship = Relationship(child_edge_node,edge_r_name,t_node,source="hetionet")
		#print(child_edge_node,father_edge_node)
		f_relationship = Relationship(child_edge_node,"instanceOf",father_edge_node,source="hetionet")
		node_ls = [s_node,child_edge_node,t_node,father_edge_node]
		edge_ls = [l_relationship,r_relationship,f_relationship]
		subgraph = Subgraph(node_ls,edge_ls)
		tx = graph.begin()
		tx.create(subgraph)
		graph.commit(tx)
	node_dict = load_nodes(node_file)
	print("loading edges...")
	num_file = sum([1 for i in open(edge_file,"r")])-1
	with open(edge_file, "r") as f:
		for line in tqdm(f.readlines()[1:],total=num_file):
			source, metaedge, target = line.strip("\n").split("\t")
			if metaedge in metaedge_dict and source in node_dict and target in node_dict:
				s_node_id,s_node_kind,s_node_name = node_dict[source]["id"],node_dict[source]["kind"],\
				node_dict[source]["name"]
				t_node_id,t_node_kind,t_node_name = node_dict[target]["id"],node_dict[target]["kind"],\
				node_dict[target]["name"]
			
				s_node = matcher.match(s_node_kind,id=s_node_id).first()
				t_node = matcher.match(t_node_kind,id=t_node_id).first()
				if not s_node:
					error_f.write(line)
					continue
				if not t_node:
					error_f.write(line)
					continue
				create_subgraph(graph,matcher,metaedge)

def main(node_file,edge_file):
	load_edges(graph,matcher,node_file,edge_file)
	print("done")

if __name__ == "__main__":
	node_csv = sys.argv[1]
	edge_csv = sys.argv[2]
	load_nodes_from_tsv(drugbank_file,metaboliteIDmapping_file)
	error_csv = node_csv.split(".")[0]+".compound.error_log"
	error_f = open(error_csv, "w")
	main(node_csv, edge_csv)
	error_f.close()