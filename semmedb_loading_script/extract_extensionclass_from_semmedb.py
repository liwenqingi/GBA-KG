# -*- coding:utf-8 -*-
#This script use to extract use class from semmedb

from pdb import pm
import sys


'''
use_class_dict = {"bact":"microbe","bpoc":"anatomy","cell":"anatomy","tisu":"anatomy",\
    "antb":"compound","clnd":"compound","dsyn":"disease","gngm":"gene"}
use_predicate_dict = {"INTERACTS_WITH":{"antb":"antb","antb":"bact","bact":"antb","clnd":"gngm","gngm":"clnd"},\
    "ASSOCIATED_WITH":{"antb":"dsyn","bact":"dsyn","dsyn":"bact","dsyn":"dsyn","dsyn":"gngm","gngm":"dsyn"},\
    "COEXISTS_WITH":{"dsyn":"dsyn"},\
    "CAUSES":{"bact":"dsyn","antb":"dsyn","gngm":"dsyn"},\
    "LOCATION_OF":{"bpoc":"dsyn","bpoc":"bact","tisu":"bact","tisu":"dsyn"},\
    "INHIBITS":{"antb":"gngm","gngm":"gngm"},\
    "AFFECTS":{"antb":"dsyn","gngm":"dsyn"},\
    "STIMULATES":{"antb":"gngm","clnd":"gngm","gngm":"gngm"},\
    "TREATS":{"antb":"bact","antb":"dsyn","clnd":"bact","clnd":"dsyn","gngm":"dsyn"},\
    "PRODUCES":{"bact":"antb","bpoc":"antb","tisu":"antb"}}
'''
#loading umls sematic file
def load_umls_semantic_dict():
    semantic_group = "/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/metamap_semantic_group.txt"
    semantic_type = "/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/metamap_semantic_type.txt"
    semantic_matching_dict = {}
    with open(semantic_type,"r") as f,open(semantic_group,"r") as F:
        for line in f.readlines():
            abbrev,code,desc = line.rstrip("\n").split("|",maxsplit=2)
            if not code in semantic_matching_dict:
                semantic_matching_dict[code] = {}
                semantic_matching_dict[code]["description"] = desc
                semantic_matching_dict[code]["abbrev"] = abbrev
        for line in F.readlines():
            _,group,code,desc = line.rstrip("\n").split("|",maxsplit=3)
            if code in semantic_matching_dict:
                semantic_matching_dict[code]["group"] = group
    semantic_dict = {}
    for key in semantic_matching_dict:
        desc,abbrev,group = semantic_matching_dict[key]["description"],semantic_matching_dict[key]["abbrev"],\
            semantic_matching_dict[key]["group"]
        if not group in semantic_dict:
            semantic_dict[abbrev] = {}
            semantic_dict[abbrev]["description"],semantic_dict[abbrev]["code"],semantic_dict[abbrev]["group"] = \
                desc,key,group
    return semantic_dict

#use umls class,class in metamap semantic group
def pymetamap_candidate_settings():
    total_use_group = set(("Anatomy","Chemicals & Drugs","Disorders","Physiology","Genes & Molecular Sequences","Living Beings"))
    semantic_dict = load_umls_semantic_dict()
    candidate_class = set()
    for k in semantic_dict:
        semantic_group = semantic_dict[k]["group"]
        if semantic_group in total_use_group:
            candidate_class.add(k)
    return candidate_class

if __name__ == "__main__":
    #PRIDICTION file from ncbi semmedb https://lhncbc.nlm.nih.gov/ii/tools/SemRep_SemMedDB_SKR/SemMedDB_download.html
    PREDICTION_CSV = "/share/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/3_mysql_load/PREDICATION.csv"
    output_csv = "./extension_predicate_class.csv"

    use_class_set = pymetamap_candidate_settings()
    with open(PREDICTION_CSV,"r") as f,open(output_csv,"w") as o:
        try:
            line = f.readline()
            while line:
                lines = line.rstrip("\n").split(",")
                predication_id,sentence_id,pmid,predicate,s_cui,s_name,s_semtype,_,\
                    o_cui,o_name,o_semtype = lines[:11]
                #replace NEG_ 
                predicate = predicate.replace("NEG_","")
                if s_semtype in use_class_set and o_semtype in use_class_set:
                    o.write(line)
                line = f.readline()
        except:
            pass
            
