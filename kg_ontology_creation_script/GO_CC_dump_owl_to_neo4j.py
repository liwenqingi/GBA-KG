# -*- coding:utf-8 -*-

import sys
import py2neo
import numpy as np
import pandas as pd
from tqdm import tqdm
from py2neo import Node, Graph, NodeMatcher, Relationship,RelationshipMatcher

def load_csv(target_file):
	df = pd.read_csv(target_file,sep="\t")
	#some settings,remoce unnamed columns from csv which resulted from unknown reasons.
	col_names = df.columns.values.copy()
	df.dropna(how="all",inplace=True, axis=1)
	df.columns = np.hstack((col_names[:-5],col_names[-2:]))
	df.fillna("None", inplace=True)
	tmp_df = df["id/alternativeId"].str.split(",",expand=True)
	df["id"],df["alternativeId"] = tmp_df[0],tmp_df[1]
	df["class"] = "Cellular component"
	return df

if __name__ == "__main__":
	target_file = sys.argv[1]
	target_err_log = target_file + ".error_log"

	target_err_f = open(target_err_log, "w")

	target_df = load_csv(target_file)
	graph = Graph("http://172.16.100.105:7474", auth=("neo4j","123"))
	#delete constraint
	#graph.schema.drop_uniqueness_constraint("Cellular component", "id")
	try:
		graph.schema.create_uniqueness_constraint("Cellular component", "id")
	except:
		pass

	node_matcher = NodeMatcher(graph)
	rel_matcher = RelationshipMatcher(graph)
	
	for i, j in tqdm(target_df.iterrows()):
		n_class, n_id, n_alternativeId, n_label, n_desc, n_crossRef, n_url, n_broadSynonym,n_exactSynonym,\
		n_narrowSynonym, n_relatedSynonym = j["class"], j["id"], j["alternativeId"],j["label"],\
		j["description"], j["crossRef"], j["url"], j["broadSynonym"], j["exactSynonym"],j["narrowSynonym"],\
		j["relatedSynonym"]
		if not node_matcher.match(n_class,id=n_id):
			target_node = Node(n_class, id=n_id, alternativeId=n_alternativeId,name=n_label,description=n_desc,\
				crossReference=n_crossRef,url=n_url,broadSynonym=n_broadSynonym,exactSynonym=n_exactSynonym,\
				narrowSynonym=n_narrowSynonym,relatedSynonym=n_relatedSynonym)
			tx = graph.begin()
			tx.create(target_node)
			graph.commit(tx)
	
	for i,j in tqdm(target_df.iterrows()):
		n_class,n_id, n_subClassOfId = j["class"], j["id"], j["subClassOfId"]
		if n_subClassOfId != "None":
			s_node = node_matcher.match(n_class, id=n_id).first()
			for each in n_subClassOfId.split(",")[:-1]:
				o_node = node_matcher.match(n_class, id=each).first()
				if s_node and o_node:
					p_relationship_name = "subClassOf"
					p_relationship = Relationship(s_node, p_relationship_name, o_node, source="GO")
					subgraph = s_node | p_relationship | o_node
					tx = graph.begin()
					tx.merge(subgraph,n_class,"id")
					graph.commit(tx)
				else:
					target_err_f.write(str(j.to_dict())+"\n")
	target_err_f.close()
	
