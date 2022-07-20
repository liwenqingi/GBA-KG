# -*- coding:utf-8 -*-
#This script help to create demo relationship into neo4j,but \
# there may be conflicts between pre-defined relationships and semmedB relationships
import sys
import py2neo
import pandas as ps
from py2neo import *

if __name__ == "__main__":
	graph = Graph("http://localhost:7474", auth=("neo4j","123"))
	matcher = NodeMatcher(graph)
	#delete constraint
	#graph.schema.drop_uniqueness_constraint("Cellular component", "id")
	protege_sub_classs = ['Gene regulates gene','Gene interacts gene','Gene covaries gene','Disease upregulates gene',\
	'Anatomy upregulates gene','Anatomy downregulates gene','Anatomy expresses gene','Disease associates gene',\
	'Disease downregulates gene','Disease upregulates gene','Compound binds gene','Compound downregulates gene',\
	'Compound upregulates gene','Gene participates biological process','Disease localizes anatomy','Disease resembles disease',\
	'Compound treasts disease','Compound resembles compound']

	try:
		graph.schema.create_uniqueness_constraint("Meta Class Relationship", "name")
	except:
		pass
	class_father_node = Node("Meta Class Relationship", name="relationship",source="Protege")
	tx = graph.begin()
	tx.merge(class_father_node,"Meta Class Relationship","name")
	graph.commit(tx)
	for j in protege_sub_classs:
		class_node = Node("Meta Class Relationship", name=j,source="Protege")
		
		tx = graph.begin()
		tx.merge(class_node,"Meta Class Relationship","name")
		graph.commit(tx)
		