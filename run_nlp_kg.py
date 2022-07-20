# -*- coding:utf-8 -*-
#author:liwenqing
#Automatic extract information from abstracts.csv which generated from generate_kg_pubmed_abstracts.py
#This script require several file
#1.semantic type file url:https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemanticTypes_2018AB.txt
#2.semantic group file url:https://lhncbc.nlm.nih.gov/ii/tools/MetaMap/Docs/SemGroups_2018.txt
#3.MRCONSO.RRF url:/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/MRCONSO.RRF
#4.disease_xref url:/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/HumanDO.owl.map
#pretrained finetuning biobert url:/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/2_test_biobert/data/out/params_biobert_model_25_epoch.pkl

import os
import sys

path = "./gba_nlp_script"

if __name__ == "__main__":
    #input_abstracts_file like /home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/kg_pubmed_abstracts.csv
    input_abstracts_file = sys.argv[1]
    output_file = sys.argv[2]

    os.system("python %s/ner_kg_text.py %s %s"%(path, input_abstracts_file, output_file))
    #Choose whether to train the model or not 
    if 1 != 1:
        os.system("python %s/test_train_biobert_cuda.py"%path)
    os.system("python %s/test_biobert_predict.py %s"%(path,input_abstracts_file+".ner"))
    os.system("python %s/ontology_mapper.py %s"%(path, input_abstracts_file+".ner.predict"))
