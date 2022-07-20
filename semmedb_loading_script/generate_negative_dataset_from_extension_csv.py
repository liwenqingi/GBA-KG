# -*- coding:utf-8 -*-
#This script use to generate negative dataset from SENTENCE.csv and negative spo tripples.

import re
from collections import defaultdict


if __name__ == "__main__":
    filtered_csv = "./extension_predicate_class.csv.filtered_pmid.csv.negative_relation.csv"
    SENTENCE_CSV = "/share/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/3_mysql_load/SENTENCE.csv"
    output_csv = filtered_csv + ".combine_text"
    
    rel_output_csv = output_csv + ".relation.csv"

    sents_dict = defaultdict(dict)
    
    rel_dict = defaultdict(int)
    #reading
    with open(filtered_csv,"r") as f,open(SENTENCE_CSV,"r") as F,open(output_csv,"w") as o,open(rel_output_csv,"w") as O:
        o.write("sentence,relation,head,head_offset,head_type,tail,tail_offset,tail_type\n")
        line = f.readline()
        while line:
            lines = line.rstrip("\n").split(",")
            group,predication_id,sentence_id,pmid,predicate,s_cui,s_name,s_semtype,_,\
                    o_cui,o_name,o_semtype = lines[:12]
            sents_dict[sentence_id]["s"],sents_dict[sentence_id]["o"] = [],[]
            sents_dict[sentence_id]["s_cui"],sents_dict[sentence_id]["o_cui"] = [],[]
            s_group,m_predicate,o_group = group.split("_")
            o_group = o_group[:-1]
            sents_dict[sentence_id]["s_group"] = s_group
            sents_dict[sentence_id]["o_group"] = o_group
            for each_s_cui,each_s_name in zip(s_cui.split("|"),s_name.split("|")):
                sents_dict[sentence_id]["s"].append(each_s_name)
                sents_dict[sentence_id]["s_cui"].append(each_s_cui)
            for each_o_cui,each_o_name in zip(o_cui.split("|"),o_name.split("|")):
                sents_dict[sentence_id]["o"].append(each_o_name)
                sents_dict[sentence_id]["o_cui"].append(each_o_cui)
            sents_dict[sentence_id]["predicate"] = m_predicate
            sents_dict[sentence_id]["s_type"],sents_dict[sentence_id]["o_type"] = s_semtype,o_semtype
            line = f.readline()
        Line = F.readline()
        while Line:
            Lines = Line.rstrip().split("\t",maxsplit=6)
            sents_id,pmid,txt_type,sents_loc,sents_start_idx,sents,sents_end_idx = Lines
            if sents_id in sents_dict:
                for s_name,o_name,s_cui,o_cui in zip(sents_dict[sents_id]["s"],sents_dict[sents_id]["o"],\
                    sents_dict[sents_id]["s_cui"],sents_dict[sents_id]["o_cui"]):
                    s_idx = sents.find(s_name)
                    o_idx = sents.find(o_name)
                    #require subject and object in sentence 
                    if s_idx!=-1 and o_idx!=-1:
                        predicate = sents_dict[sents_id]["predicate"]
                        s_group,o_group = sents_dict[sents_id]["s_group"],sents_dict[sents_id]["o_group"]
                        #count negative spo tripples numbers.
                        rel_dict[(s_group,o_group,predicate)] += 1
                        o.write(",".join([sents,predicate,s_name,str(s_idx),s_group,o_name,str(o_idx),o_group])+"\n")
            Line = F.readline()
        for k,v in rel_dict.items():
            O.write(",".join(k)+"\t"+str(v)+"\n")