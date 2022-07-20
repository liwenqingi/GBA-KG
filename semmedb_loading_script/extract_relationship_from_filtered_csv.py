# -*- coding:utf-8 -*-
#This script use to extract positive dataset from extension_predicate_class.csv.filtered_pmid.csv

import sys
from collections import defaultdict

#candidate positive class settings
bact_loc_anatomy = {"subject":set(("cell","bpoc","tisu","emst","anst","ffas","bdsu")),\
    "predicate":set(("LOCATION_OF","NEG_LOCATION_OF")),"object":set(["bact","fngs","arch"])}
disease_loc_anatomy = {"subject":set(("cell","bpoc","tisu","emst","anst","ffas","bdsu")),\
    "predicate":set(("LOCATION_OF","NEG_LOCATION_OF")),"object":set(("acab","anab","comd",\
        "cgab","dsyn","emod","fndg","inpo","mobd","neop","patf","sosy"))}
gene_interacts_anatomy = {"subject":set(["gngm"]),"predicate":set(["INTERACTS_WITH"]),"object":\
    set(["cell"])}
compound_loc_anatomy = {"subject":set(("cell","bpoc","tisu","emst","anst","ffas","bdsu")),\
    "predicate":set(("LOCATION_OF","NEG_LOCATION_OF")),"object":set(("aapp","antb","bacs","bodm",\
        "chem","chvf","chvs","clnd","elii","enzy","hops","horm","imft","irda","inch","nnon","orch",\
            "phsu","rcpt","vita"))}
gene_stimulates_gene = {"subject":set(["gngm"]),"predicate":set(("STIMULATES","NEG_STIMULATES")),\
    "object":set(["gngm"])}
gene_inhibits_gene = {"subject":set(["gngm"]),"predicate":set(("INHIBITS","NEG_INHIBITS")),\
    "object":set(["gngm"])}
gene_interacts_gene = {"subject":set(["gngm"]),"predicate":set(("INTERACTS_WITH","NEG_INTERACTS_WITH")),\
    "object":set(["gngm"])}
gene_associates_disease = {"subject":set(["gngm"]),"predicate":set(("ASSOCIATED_WITH","NEG_ASSOCIATED_WITH",\
    "CAUSES","PREDISPOSES","AFFECTS","TREATS","NEG_TREATS")),"object":set(("acab","anab","comd","cgab","dsyn","emod","fndg","inpo",\
        "mobd","neop","patf","sosy"))}
gene_participates_physiology = {"subject":set(["gngm"]),"predicate":set(("CAUSES","AFFECTS","DISRUPTS")),\
    "object":set(("celf","orgf","ortf","phsf","clna","genf","menp","moft","orga"))}
compound_stimulates_gene = {"subject":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita")),"predicate":set(["STIMULATES"]),\
        "object":set(["gngm"])}
compound_inhibits_gene = {"subject":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita")),"predicate":set(["INHIBITS"]),\
        "object":set(["gngm"])}
compound_interacts_gene = {"subject":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita")),"predicate":set(["INTERACTS_WITH","NEG_INTERACTS_WITH"]),\
        "object":set(["gngm"])}
gene_interacts_compound = {"object":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita")),"predicate":set(["INTERACTS_WITH","NEG_INTERACTS_WITH"]),\
        "subject":set(["gngm"])}
disease_resembles_disease = {"subject":set(("acab","anab","comd","cgab","dsyn","emod","fndg","inpo","mobd",\
    "neop","patf","sosy")),"predicate":set(["COEXISTS_WITH"]),"object":set(("acab","anab","comd","cgab","dsyn",\
        "emod","fndg","inpo","mobd","neop","patf","sosy"))}
compound_treats_disease = {"subject":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita")),"predicate":set(["TREATS"]),"object":\
        set(("acab","anab","comd","cgab","dsyn","emod","fndg","inpo","mobd","neop","patf","sosy"))}
compound_causes_disease = {"subject":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita")),"predicate":set(["CAUSES"]),"object":\
        set(("acab","anab","comd","cgab","dsyn","emod","fndg","inpo","mobd","neop","patf","sosy"))}
bact_associates_disease = {"subject":set(["bact","fngs","arch"]),"predicate":set(("CAUSES","ASSOCIATED_WITH","NEG_CAUSES",\
    "NEG_ASSOCIATED_WITH")),"object":set(("acab","anab","comd","cgab","dsyn","emod","fndg","inpo","mobd","neop",\
        "patf","sosy"))}
compound_resembles_compound = {"subject":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita")),"predicate":set(["ISA"]),"object":\
        set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy","hops","horm","imft","irda","inch",\
            "nnon","orch","phsu","rcpt","vita"))}
bact_produces_compound = {"subject":set(["bact","fngs","arch"]),"predicate":set(("PRODUCES","LOCATION_OF")),"object":set(("aapp","antb",\
    "bacs","bodm","chem","chvf","chvs","clnd","elii","enzy","hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita"))}
    
if __name__ == "__main__":
    predicate_csv = sys.argv[1]
    output_csv = predicate_csv + ".selected_relation.csv"
    relation_csv = output_csv + ".relation_count"

    relation_dict = defaultdict(int)
    #reading 
    with open(predicate_csv,"r") as f, open(output_csv, "w") as o,open(relation_csv,"w") as O:
        line = f.readline()
        while line:
            lines = line.rstrip("\n").split(",")
            predication_id,sentence_id,pmid,predicate,s_cui,s_name,s_semtype,_,\
                    o_cui,o_name,o_semtype = lines[:11]
            #subject type and object type and predicate type in candidate semmedb 
            if s_semtype in bact_loc_anatomy["subject"] and o_semtype in bact_loc_anatomy["object"] and predicate in bact_loc_anatomy["predicate"]:
                lines[5],lines[9],lines[6],lines[10],lines[4],lines[8] = lines[9],lines[5],lines[10],lines[6],lines[8],lines[4]
                relation_dict["bact_localizes_anatomy"] += 1
                o.write("bact_localizes_anatomy," + ",".join(lines)+"\n")
            elif s_semtype in disease_loc_anatomy["subject"] and o_semtype in disease_loc_anatomy["object"] and predicate in disease_loc_anatomy["predicate"]:
                lines[5],lines[9],lines[6],lines[10],lines[4],lines[8] = lines[9],lines[5],lines[10],lines[6],lines[8],lines[4]
                relation_dict["disease_localizes_anatomy"] += 1
                o.write("disease_localizes_anatomy," + ",".join(lines) + "\n")
            elif s_semtype in gene_interacts_anatomy["subject"] and o_semtype in gene_interacts_anatomy["object"] and predicate in gene_interacts_anatomy["predicate"]:
                relation_dict["gene_interacts_anatomy"] += 1
                o.write("gene_interacts_anatomy," + ",".join(lines) + "\n")
            elif s_semtype in compound_loc_anatomy["subject"] and o_semtype in compound_loc_anatomy["object"] and predicate in compound_loc_anatomy["predicate"]:
                lines[5],lines[9],lines[6],lines[10],lines[4],lines[8] = lines[9],lines[5],lines[10],lines[6],lines[8],lines[4]
                relation_dict["compound_localizes_anatomy"] += 1
                o.write("compound_localizes_anatomy," + ",".join(lines) + "\n")        
            elif s_semtype in gene_stimulates_gene["subject"] and o_semtype in gene_stimulates_gene["object"] and predicate in gene_stimulates_gene["predicate"]:
                relation_dict["gene_stimulates_gene"] += 1
                o.write("gene_stimulates_gene," + ",".join(lines) + "\n")  
            elif s_semtype in gene_inhibits_gene["subject"] and o_semtype in gene_inhibits_gene["object"] and predicate in gene_inhibits_gene["predicate"]:
                relation_dict["gene_inhibits_gene"] += 1
                o.write("gene_inhibits_gene," + ",".join(lines) + "\n")          
            elif s_semtype in gene_interacts_gene["subject"] and o_semtype in gene_interacts_gene["object"] and predicate in gene_interacts_gene["predicate"]:
                relation_dict["gene_interacts_gene"] += 1
                o.write("gene_interacts_gene," + ",".join(lines) + "\n")          
            elif s_semtype in gene_associates_disease["subject"] and o_semtype in gene_associates_disease["object"] and predicate in gene_associates_disease["predicate"]:
                relation_dict["gene_associates_disease"] += 1
                o.write("gene_associates_disease," + ",".join(lines) + "\n") 
            elif s_semtype in gene_participates_physiology["subject"] and o_semtype in gene_participates_physiology["object"] and predicate in gene_participates_physiology["predicate"]:
                relation_dict["gene_participates_physiology"] += 1
                o.write("gene_participates_physiology," + ",".join(lines) + "\n") 
            elif s_semtype in compound_stimulates_gene["subject"] and o_semtype in compound_stimulates_gene["object"] and predicate in compound_stimulates_gene["predicate"]:
                relation_dict["compound_stimulates_gene"] += 1
                o.write("compound_stimulates_gene," + ",".join(lines) + "\n") 
            elif s_semtype in compound_inhibits_gene["subject"] and o_semtype in compound_inhibits_gene["object"] and predicate in compound_inhibits_gene["predicate"]:
                relation_dict["compound_inhibits_gene"] += 1
                o.write("compound_inhibits_gene," + ",".join(lines) + "\n")
            elif s_semtype in compound_interacts_gene["subject"] and o_semtype in compound_interacts_gene["object"] and predicate in compound_interacts_gene["predicate"]:
                relation_dict["compound_interacts_gene"] += 1
                o.write("compound_interacts_gene," + ",".join(lines) + "\n")
            elif s_semtype in gene_interacts_compound["subject"] and o_semtype in gene_interacts_compound["object"] and predicate in gene_interacts_compound["predicate"]:
                lines[5],lines[9],lines[6],lines[10],lines[4],lines[8] = lines[9],lines[5],lines[10],lines[6],lines[8],lines[4]
                relation_dict["compound_interacts_gene"] += 1
                o.write("compound_interacts_gene," + ",".join(lines) + "\n")
            
            elif s_semtype in disease_resembles_disease["subject"] and o_semtype in disease_resembles_disease["object"] and predicate in disease_resembles_disease["predicate"]:
                relation_dict["disease_resembles_disease"] += 1
                o.write("disease_resembles_disease," + ",".join(lines) + "\n")
            elif s_semtype in compound_treats_disease["subject"] and o_semtype in compound_treats_disease["object"] and predicate in compound_treats_disease["predicate"]:
                relation_dict["compound_treats_disease"] += 1
                o.write("compound_treats_disease," + ",".join(lines) + "\n")
            elif s_semtype in compound_causes_disease["subject"] and o_semtype in compound_causes_disease["object"] and predicate in compound_causes_disease["predicate"]:
                relation_dict["compound_causes_disease"] += 1
                o.write("compound_causes_disease," + ",".join(lines) + "\n")
            elif s_semtype in bact_associates_disease["subject"] and o_semtype in bact_associates_disease["object"] and predicate in bact_associates_disease["predicate"]:
                relation_dict["bact_associates_disease"] += 1
                o.write("bact_associates_disease," + ",".join(lines) + "\n")
            elif s_semtype in compound_resembles_compound["subject"] and o_semtype in compound_resembles_compound["object"] and predicate in compound_resembles_compound["predicate"]:
                relation_dict["compound_resembles_compound"] += 1
                o.write("compound_resembles_compound," + ",".join(lines) + "\n")
            elif s_semtype in bact_produces_compound["subject"] and o_semtype in bact_produces_compound["object"] and predicate in bact_produces_compound["predicate"]:
                relation_dict["bact_produces_compound"] += 1
                o.write("bact_produces_compound," + ",".join(lines) + "\n")
            line = f.readline()
        for k,v in relation_dict.items():
            O.write("\t".join([k,str(v)])+"\n")