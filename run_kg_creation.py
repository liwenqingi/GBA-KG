# -*- coding:utf-8 -*-
#create knowledge graph from ontology,load hetionet as pre-defined base knowledge graph
#This script require several file,ontology file can search from ebi ontology web.
#1.OGG Ontology file 
#2.Human Disease Ontology file
#3.CHEBI ontology file
#4.Gene Ontology file
#5.Uberon ontology file
#6.nodes.dmp from ncbi
#7.hetionet nodes and edges file https://github.com/hetio/hetionet
#before running this script,you should start neo4j first.Running neo4j on linux can follow \
#/share/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/3_demo2_ncbitaxon_owl_neo4j_3.5,how to build \
# neo4j docker image,can follow /share/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/Dockerfile. 
#A ready-made Neo4j graph.db containing all entities and hetionet is available,\
# /share/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/3_demo2_ncbitaxon_owl_neo4j_3.5/graph.db

import os
import sys


path = "./kg_ontology_creation_script"
ontology_path = "/home/liwenqing/liwenqing_hpc/1_xbiome/5_ontology/"

if __name__ == "__main__":
      #Compound
      os.system("python %s/CHEBI_loadcsv_from_rdflib.py %s/chebi.owl"%(path,ontology_path))
      os.system("python %s/CHEBI_dump_owl_to_neo4j.py %s/chebi.owl.csv"%(path,ontology_path))
      #Gene
      os.system("python %s/OGG_loadcsv_from_rdflib.py %s/ogg.owl"%(path,ontology_path))
      os.system("python %s/OGG_dump_owl_to_neo4j.py %s/ogg.owl.csv"%(path,ontology_path))
      #MF,BP,CC from GO
      os.system("python %s/GO_loadcsv_from_rdflib.py %s/go.owl"%(path,ontology_path))
      os.system("python %s/GO_MF_dump_owl_to_neo4j.py %s/go.owl.molecular_function.csv"%(path,ontology_path))
      os.system("python %s/GO_BP_dump_owl_to_neo4j.py %s/go.owl.biological_process.csv"%(path,ontology_path))
      os.system("python %s/GO_CC_dump_owl_to_neo4j.py %s/go.owl.cellular_component.csv"%(path,ontology_path))
      #Anatomy
      os.system("python %s/UBERON_loadcsv_from_rdflib.py %s/uberon.owl"%(path,ontology_path))
      os.system("python %s/UBERON_dump_owl_to_neo4j.py %s/uberon.owl.csv"%(path,ontology_path))
      #Disease
      os.system("python %s/DO_loadcsv_from_rdflib.py %s/HumanDO.owl"%(path,ontology_path))
      os.system("python %s/DO_dump_owl_to_neo4j.py %s/HumanDO.owl.csv"%(path,ontology_path))
      
      #Bacteria
      os.system("python %s/microbe_ncbitaxon_to_neo4j_format.py \
            %s/1_ncbitaxon/bacteria.txt %s/1_ncbitaxon/nodes.dmp"%(path,ontology_path,ontology_path))
      os.system("python %s/MICROBE_dump_owl_to_neo4j.py %s/1_ncbitaxon/bacteria.txt.neo4j"%(path,ontology_path))
      
      #create metaclass relationship
      os.system("python %s/create_demo_metaclass_relationship.py"%path)
      #loading hetionet
      os.system("python %s/1_hetionet_load_csv_to_neo4j.py %s/3_hetionet_source/nodes.tsv %s/3_hetionet_source/edges.sif"%\
            (path,ontology_path,ontology_path))
      os.system("python %s/3_hetionet_compound_load_csv_to_neo4j.py %s/3_hetionet_source/nodes.tsv \
            %s/3_hetionet_source/edges.sif"%(path,ontology_path,ontology_path))

