# -*- coding:utf-8 -*-
#This script help to extract bacteria information from taxonkit result file and nodes.dmp from ncbi
#taxonkit res file:/home/liwenqing/liwenqing_hpc/1_xbiome/5_ontology/1_ncbitaxon/bacteria.txt
#nodes.dmp:/home/liwenqing/liwenqing_hpc/1_xbiome/5_ontology/1_ncbitaxon/nodes.dmp

import sys

if __name__ == "__main__":
	bacteria_archaea_file = sys.argv[1]
	nodes_dump_file = sys.argv[2]
	bacteria_output_file = bacteria_archaea_file + ".neo4j"

	url_prefix = "http://purl.obolibrary.org/obo/NCBITaxon_"

	bacteria_dict = {}
	with open(bacteria_archaea_file, "r") as f:
		for line in f.readlines():
			lines = line.strip().split("\t")
			lines = lines[0].split(" ")
			ncbi_id,name = lines[0]," ".join(lines[1:])
			bacteria_dict[ncbi_id] = name
	organism_rank_dict = {}
	with open(nodes_dump_file,"r") as f:
		for line in f.readlines():
			lines = line.strip().split("\t")
			self_id,father_id, rank = lines[0],lines[2],lines[4]
			if not self_id in organism_rank_dict:
				organism_rank_dict[self_id] = {}
			organism_rank_dict[self_id]["subclassOf"] = url_prefix +father_id
			organism_rank_dict[self_id]["rank"] = rank

	with open(bacteria_output_file,"w") as o:
		for taxon_id in bacteria_dict:
			if not taxon_id in organism_rank_dict:
				print(taxon_id)
			else:
				o.write("id|"+taxon_id+"\tname|"+bacteria_dict[taxon_id]+"\turl|"+url_prefix+taxon_id+"\t")
				for k,v in organism_rank_dict[taxon_id].items():
					o.write(k+"|"+v+"\t")
				o.write("\n")