# -*- coding:utf-8 -*-
#This script use to extract PRIDCTION.csv with pmid sentence in SENTENCE.csv

import sys

if __name__ == "__main__":
    #SENTENCE.csv from ncbi semmedb https://lhncbc.nlm.nih.gov/ii/tools/SemRep_SemMedDB_SKR/SemMedDB_download.html
    use_class_csv = "./extension_predicate_class.csv"
    SENTENCE_CSV = "/share/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/3_mysql_load/SENTENCE.csv"
    output_csv = use_class_csv + ".filtered_pmid.csv"
    #negative_total_csv = "/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/3_mysql_load/total_negative_sample.csv"
    #negative_total_output_csv = negative_total_csv + ".filtered_pmid.csv"
   

    pmid_set = set()
    #nonsense_entity_set = set(("Genes","Disease","Antibiotics","Bacteria"))
    #with open(use_class_csv,"r") as f,open(SENTENCE_CSV,"r") as F,open(negative_total_csv,"r") as f1,\
    #    open(output_csv,"w") as o,open(negative_total_output_csv,"w") as o1:
    with open(use_class_csv,"r") as f,open(SENTENCE_CSV,"r") as F,open(output_csv,"w") as o:
        Line = F.readline()
        while Line:
            Lines = Line.rstrip().split("\t")
            pmid_set.add(Lines[1])
            Line = F.readline()
        line = f.readline()
        while line:
            lines = line.rstrip("\n").split(",")
            predication_id,sentence_id,pmid,predicate,s_cui,s_name,s_semtype,_,\
                    o_cui,o_name,o_semtype = lines[:11]
            #if pmid in pmid_set and not s_semtype == "clnd" and not o_semtype == "clnd" and \
            #    not o_name in nonsense_entity_set and not s_name in nonsense_entity_set:
            if pmid in pmid_set:
                o.write(line)
            line = f.readline()
       