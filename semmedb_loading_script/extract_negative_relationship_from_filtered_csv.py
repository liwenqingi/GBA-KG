# -*- coding:utf-8 -*-
#This script use to extract negative dataset from semmedb

import sys
from collections import defaultdict

#bact-others-anatomy
bact_others_anatomy1 = {"subject":set(("cell","bpoc","tisu","emst","anst","ffas","bdsu")),\
    "predicate":set(["LOCATION_OF","NEG_LOCATION_OF"]),"object":set(["bact","arch","fngs"])}
#replace bact with anatomy
bact_others_anatomy2 = {"subject":set(("cell","bpoc","tisu","emst","anst","ffas","bdsu")),\
    "predicate":set(["LOCATION_OF","NEG_LOCATION_OF"]),"object":set(("cell","bpoc","tisu","emst",\
        "anst","ffas","bdsu"))}

#disease-others-anatomy
disease_others_anatomy1 = {"subject":set(["cell","bpoc","tisu","emst","anst","ffas","bdsu"]),\
    "predicate":set(("LOCATION_OF","NEG_LOCATION_OF")),"object":set(("acab","anab","comd",\
        "cgab","dsyn","emod","fndg","inpo","mobd","neop","patf","sosy"))}
#exchange s and o
disease_others_anatomy2 = {"object":set(["cell","bpoc","tisu","emst","anst","ffas","bdsu"]),\
    "predicate":set(("LOCATION_OF","NEG_LOCATION_OF")),"subject":set(("acab","anab","comd",\
        "cgab","dsyn","emod","fndg","inpo","mobd","neop","patf","sosy"))}
#replace disease with anatomy
disease_others_anatomy3 = {"object":set(["cell","bpoc","tisu","emst","anst","ffas","bdsu","celc"]),\
    "predicate":set(("LOCATION_OF","NEG_LOCATION_OF")),"subject":set(["cell","bpoc",\
        "tisu","emst","anst","ffas","bdsu","celc"])}
#replace anatomy with disease
disease_others_anatomy4 = {"object":set(("acab","anab","comd",\
        "cgab","dsyn","emod","fndg","inpo","mobd","neop","patf","sosy")),\
    "predicate":set(("LOCATION_OF","NEG_LOCATION_OF")),"subject":set(("acab","anab","comd",\
        "cgab","dsyn","emod","fndg","inpo","mobd","neop","patf","sosy"))}

#gene-others-anatomy
gene_others_anatomy = {"subject":set(["gngm","amas","crbs","mosq","nusq"]),"predicate":set(["INTERACTS_WITH"]),"object":\
    set(["cell","bpoc","tisu","emst","anst","ffas","bdsu"])}
#repalce gene with anatomy
gene_others_anatomy2 = {"subject":set(["cell","bpoc","tisu","emst","anst","ffas","bdsu","celc"]),\
    "predicate":set(["INTERACTS_WITH","NEG_INTERACTS_WITH"]),"object":set(["cell","bpoc","tisu",\
        "emst","anst","ffas","bdsu","celc"])}
#repalce anatomy with gene
gene_others_anatomy3 = {"subject":set(["gngm","amas","crbs","mosq","nusq"]),\
    "predicate":set(["INTERACTS_WITH","NEG_INTERACTS_WITH"]),"object":set(["gngm","amas","crbs","mosq","nusq"])}


#compoound-others-anatomy
compound_others_anatomy1 = {"subject":set(("cell","bpoc","tisu","emst","anst","ffas","bdsu")),\
    "predicate":set(("LOCATION_OF","NEG_LOCATION_OF")),"object":set(("aapp","antb","bacs","bodm",\
        "chem","chvf","chvs","clnd","elii","enzy","hops","horm","imft","irda","inch","nnon","orch",\
            "phsu","rcpt","vita"))}
#exchange s and o
compound_others_anatomy2 = {"object":set(("cell","bpoc","tisu","emst","anst","ffas","bdsu")),\
    "predicate":set(("LOCATION_OF","NEG_LOCATION_OF")),"subject":set(("aapp","antb","bacs","bodm",\
        "chem","chvf","chvs","clnd","elii","enzy","hops","horm","imft","irda","inch","nnon","orch",\
            "phsu","rcpt","vita"))}
#replace compound with anatomy
compound_others_anatomy3 = {"object":set(("cell","bpoc","tisu","emst","anst","ffas","bdsu")),\
    "predicate":set(("LOCATION_OF","NEG_LOCATION_OF")),"subject":set(["cell","bpoc","tisu","emst",\
        "anst","ffas","bdsu"])}
#replace anatomy with compound
compound_others_anatomy4 = {"object":set(("aapp","antb","bacs","bodm",\
        "chem","chvf","chvs","clnd","elii","enzy","hops","horm","imft","irda","inch","nnon","orch",\
            "phsu","rcpt","vita")),\
    "predicate":set(("PART_OF","INTERACTS_WITH","PRODUCES")),"subject":set(("aapp","antb","bacs","bodm",\
        "chem","chvf","chvs","clnd","elii","enzy","hops","horm","imft","irda","inch","nnon","orch",\
            "phsu","rcpt","vita"))}

#gene-other-gene
gene_others_gene1 = {"subject":set(["gngm","amas","crbs","mosq","nusq"]),"predicate":set(("STIMULATES","NEG_STIMULATES","INHIBITS",\
    "NEG_INHIBITS","INTERACTS_WITH","NEG_INTERACTS_WITH")),"object":set(["gngm","amas","crbs","mosq","nusq"])}
#replace gene to compound
gene_others_gene2 = {"subject":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita")),"predicate":set(("STIMULATES",\
    "NEG_STIMULATES","INHIBITS","NEG_INHIBITS","INTERACTS_WITH","NEG_INTERACTS_WITH")),"object":set(["gngm","amas","crbs","mosq","nusq"])}

#gene-other-disease
gene_others_disease1 = {"subject":set(["gngm","amas","crbs","mosq","nusq"]),"predicate":set(("ASSOCIATED_WITH","NEG_ASSOCIATED_WITH",\
    "CAUSES","PREDISPOSES","AFFECTS","TREATS")),"object":set(("acab","anab","comd","cgab","dsyn","emod","fndg","inpo",\
        "mobd","neop","patf","sosy"))}
#replace gene with disease
gene_others_disease2 = {"subject":set(("acab","anab","comd","cgab","dsyn","emod","fndg","inpo","mobd","neop","patf","sosy")),\
    "predicate":set(("ASSOCIATED_WITH","NEG_ASSOCIATED_WITH",\
    "CAUSES","PREDISPOSES","AFFECTS","TREATS")),"object":set(("acab","anab","comd","cgab","dsyn","emod","fndg","inpo",\
        "mobd","neop","patf","sosy"))}
#gene-other-physiology
gene_others_physiology1 = {"subject":set(["gngm","amas","crbs","mosq","nusq"]),"predicate":set(("CAUSES","AFFECTS","DISRUPTS")),\
    "object":set(("celf","orgf","ortf","phsf","clna","genf","menp","moft","orga"))}
#exchange subject and object
gene_others_physiology2 = {"object":set(["gngm","amas","crbs","mosq","nusq"]),"predicate":set(("CAUSES","AFFECTS","DISRUPTS")),\
    "subject":set(("celf","orgf","ortf","phsf","clna","genf","menp","moft","orga"))}
#replace gene with physiology
gene_others_physiology3 = {"object":set(("celf","orgf","ortf","phsf","clna","genf","menp","moft","orga")),"predicate":set(("CAUSES",\
    "AFFECTS","DISRUPTS")),"subject":set(("celf","orgf","ortf","phsf","clna","genf","menp","moft","orga"))}

#compound-other-gene
compound_others_gene1 = {"subject":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita")),"predicate":set(["STIMULATES","INHIBITS",\
        "INTERACTS_WITH","NEG_INTERACTS_WITH"]),"object":set(["gngm","amas","crbs","mosq","nusq"])}
#exchange s and o
compound_others_gene2 = {"object":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita")),"predicate":set(["STIMULATES","INHIBITS",\
        "INTERACTS_WITH","NEG_INTERACTS_WITH"]),"subject":set(["gngm","amas","crbs","mosq","nusq"])}
#replace gene with compound
compound_others_gene3 = {"object":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita")),"predicate":set(["STIMULATES","INHIBITS",\
        "INTERACTS_WITH","NEG_INTERACTS_WITH"]),"subject":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita"))}

#disease-others-disease
disease_others_disease1 = {"subject":set(("acab","anab","comd","cgab","dsyn","emod","fndg","inpo","mobd",\
    "neop","patf","sosy")),"predicate":set(["COEXISTS_WITH"]),"object":set(("acab","anab","comd","cgab","dsyn",\
        "emod","fndg","inpo","mobd","neop","patf","sosy"))}
#repalce disease with compound
disease_others_disease2 = {"subject":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita")),"predicate":set(["COEXISTS_WITH"]),\
        "object":set(("acab","anab","comd","cgab","dsyn","emod","fndg","inpo","mobd","neop","patf","sosy"))}

#compound-others-disease
compound_others_disease1 = {"subject":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita")),"predicate":set(["TREATS","CAUSES","PREDISPOSES",\
        ]),"object":set(("acab","anab","comd","cgab","dsyn","emod","fndg","inpo","mobd","neop","patf","sosy"))}

#bact-others-disease
bact_others_disease1 = {"subject":set(["bact","arch","fngs"]),"predicate":set(("CAUSES","ASSOCIATED_WITH","NEG_CAUSES",\
    "NEG_ASSOCIATED_WITH")),"object":set(("acab","anab","comd","cgab","dsyn","emod","fndg","inpo","mobd","neop",\
        "patf","sosy"))}
#exchange s and o
bact_others_disease2 = {"object":set(["bact","arch","fngs"]),"predicate":set(("CAUSES","ASSOCIATED_WITH","NEG_CAUSES",\
    "NEG_ASSOCIATED_WITH")),"subject":set(("acab","anab","comd","cgab","dsyn","emod","fndg","inpo","mobd","neop",\
        "patf","sosy"))}
#replace bact with disease
bact_others_disease3 = {"object":set(("acab","anab","comd","cgab","dsyn","emod","fndg","inpo","mobd","neop",\
        "patf","sosy")),"predicate":set(("CAUSES","ASSOCIATED_WITH","NEG_CAUSES",\
    "NEG_ASSOCIATED_WITH")),"subject":set(("acab","anab","comd","cgab","dsyn","emod","fndg","inpo","mobd","neop",\
        "patf","sosy"))}
#replace disease with compound
bact_others_disease4 = {"subject":set(["bact","arch","fngs"]),"predicate":set(("CAUSES","ASSOCIATED_WITH","NEG_CAUSES",\
    "NEG_ASSOCIATED_WITH")),"object":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita"))}

#compound-others-compound
compound_others_compound = {"subject":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy",\
    "hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita")),"predicate":set(["ISA"]),"object":\
        set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy","hops","horm","imft","irda","inch",\
            "nnon","orch","phsu","rcpt","vita"))}

#bact-others-compound
bact_others_compound1 = {"subject":set(["bact","arch","fngs"]),"predicate":set(("PRODUCES","LOCATION_OF")),"object":set(("aapp","antb",\
    "bacs","bodm","chem","chvf","chvs","clnd","elii","enzy","hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita"))}
#exchange s and o
bact_others_compound2 = {"object":set(["bact","arch","fngs"]),"predicate":set(("PRODUCES","LOCATION_OF")),"subject":set(("aapp","antb",\
    "bacs","bodm","chem","chvf","chvs","clnd","elii","enzy","hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita"))}
#replace bact with compound
bact_others_compound3 = {"object":set(("aapp","antb","bacs","bodm","chem","chvf","chvs","clnd","elii","enzy","hops","horm","imft","irda",\
    "inch","nnon","orch","phsu","rcpt","vita")),"predicate":set(("PRODUCES","LOCATION_OF")),"subject":set(("aapp","antb",\
    "bacs","bodm","chem","chvf","chvs","clnd","elii","enzy","hops","horm","imft","irda","inch","nnon","orch","phsu","rcpt","vita"))}

if __name__ == "__main__":
    predicate_csv = sys.argv[1]
    output_csv = predicate_csv + ".negative_relation.csv"
    relation_csv = output_csv + ".negative_relation_count"

    relation_dict = defaultdict(int)
    #reading
    with open(predicate_csv,"r") as f, open(output_csv, "w") as o,open(relation_csv,"w") as O:
        line = f.readline()
        while line:
            lines = line.rstrip("\n").split(",")
            predication_id,sentence_id,pmid,predicate,s_cui,s_name,s_semtype,_,\
                    o_cui,o_name,o_semtype = lines[:11]
            #subject type and object type in semmedb and not predicate type in candidate semmedb 
            if s_semtype in bact_others_anatomy1["subject"] and o_semtype in bact_others_anatomy1["object"] and not predicate in bact_others_anatomy1["predicate"]:
                lines[5],lines[9],lines[6],lines[10],lines[4],lines[8] = lines[9],lines[5],lines[10],lines[6],lines[8],lines[4]
                relation_dict["bact_others_anatomy1"] += 1
                o.write("bact_others_anatomy1," + ",".join(lines)+"\n")
            elif s_semtype in bact_others_anatomy2["subject"] and o_semtype in bact_others_anatomy2["object"] and not predicate in bact_others_anatomy2["predicate"]:
                relation_dict["bact_others_anatomy2"] += 1
                o.write("bact_others_anatomy2," + ",".join(lines) + "\n")

            elif s_semtype in disease_others_anatomy1["subject"] and o_semtype in disease_others_anatomy1["object"] and not predicate in disease_others_anatomy1["predicate"]:
                lines[5],lines[9],lines[6],lines[10],lines[4],lines[8] = lines[9],lines[5],lines[10],lines[6],lines[8],lines[4]
                relation_dict["disease_others_anatomy1"] += 1
                o.write("disease_others_anatomy1," + ",".join(lines) + "\n")
            elif s_semtype in disease_others_anatomy2["subject"] and o_semtype in disease_others_anatomy2["object"] and not predicate in disease_others_anatomy2["predicate"]:
                relation_dict["disease_others_anatomy2"] += 1
                o.write("disease_others_anatomy2," + ",".join(lines) + "\n")
            elif s_semtype in disease_others_anatomy3["subject"] and o_semtype in disease_others_anatomy3["object"] and not predicate in disease_others_anatomy3["predicate"]:
                relation_dict["disease_others_anatomy3"] += 1
                o.write("disease_others_anatomy3," + ",".join(lines) + "\n")
            elif s_semtype in disease_others_anatomy4["subject"] and o_semtype in disease_others_anatomy4["object"] and predicate in disease_others_anatomy4["predicate"]:
                relation_dict["disease_others_anatomy4"] += 1
                o.write("disease_others_anatomy4," + ",".join(lines) + "\n")

            elif s_semtype in gene_others_anatomy["subject"] and o_semtype in gene_others_anatomy["object"] and not predicate in gene_others_anatomy["predicate"]:
                relation_dict["gene_others_anatomy1"] += 1
                o.write("gene_others_anatomy1," + ",".join(lines) + "\n")
            elif s_semtype in gene_others_anatomy2["subject"] and o_semtype in gene_others_anatomy2["object"] and not predicate in gene_others_anatomy2["predicate"]:
                relation_dict["gene_others_anatomy2"] += 1
                o.write("gene_others_anatomy2," + ",".join(lines) + "\n")
            elif s_semtype in gene_others_anatomy3["subject"] and o_semtype in gene_others_anatomy3["object"] and not predicate in gene_others_anatomy3["predicate"]:
                relation_dict["gene_others_anatomy3"] += 1
                o.write("gene_others_anatomy3," + ",".join(lines) + "\n")

            elif s_semtype in compound_others_anatomy1["subject"] and o_semtype in compound_others_anatomy1["object"] and not predicate in compound_others_anatomy1["predicate"]:
                lines[5],lines[9],lines[6],lines[10],lines[4],lines[8] = lines[9],lines[5],lines[10],lines[6],lines[8],lines[4]
                relation_dict["compound_others_anatomy1"] += 1
                o.write("compound_others_anatomy1," + ",".join(lines) + "\n")
            elif s_semtype in compound_others_anatomy2["subject"] and o_semtype in compound_others_anatomy2["object"] and not predicate in compound_others_anatomy2["predicate"]:
                relation_dict["compound_others_anatomy2"] += 1
                o.write("compound_others_anatomy2," + ",".join(lines) + "\n")
            elif s_semtype in compound_others_anatomy3["subject"] and o_semtype in compound_others_anatomy3["object"] and not predicate in compound_others_anatomy3["predicate"]:
                relation_dict["compound_others_anatomy3"] += 1
                o.write("compound_others_anatomy3," + ",".join(lines) + "\n")
            elif s_semtype in compound_others_anatomy4["subject"] and o_semtype in compound_others_anatomy4["object"] and not predicate in compound_others_anatomy4["predicate"]:
                relation_dict["compound_others_anatomy4"] += 1
                o.write("compound_others_anatomy4," + ",".join(lines) + "\n")

            elif s_semtype in gene_others_gene1["subject"] and o_semtype in gene_others_gene1["object"] and not predicate in gene_others_gene1["predicate"]:
                relation_dict["gene_others_gene1"] += 1
                o.write("gene_others_gene1," + ",".join(lines) + "\n")
            elif s_semtype in gene_others_gene2["subject"] and o_semtype in gene_others_gene2["object"] and not predicate in gene_others_gene2["predicate"]:
                relation_dict["gene_others_gene2"] += 1
                o.write("gene_others_gene2," + ",".join(lines) + "\n")

            elif s_semtype in gene_others_disease1["subject"] and o_semtype in gene_others_disease1["object"] and not predicate in gene_others_disease1["predicate"]:
                relation_dict["gene_others_disease1"] += 1
                o.write("gene_others_disease1," + ",".join(lines) + "\n")
            elif s_semtype in gene_others_disease2["subject"] and o_semtype in gene_others_disease2["object"] and not predicate in gene_others_disease2["predicate"]:
                relation_dict["gene_others_disease2"] += 1
                o.write("gene_others_disease2," + ",".join(lines) + "\n")
            
            elif s_semtype in gene_others_physiology1["subject"] and o_semtype in gene_others_physiology1["object"] and not predicate in gene_others_physiology1["predicate"]:
                relation_dict["gene_others_physiology1"] += 1
                o.write("gene_others_physiology1," + ",".join(lines) + "\n")
            elif s_semtype in gene_others_physiology2["subject"] and o_semtype in gene_others_physiology2["object"] and not predicate in gene_others_physiology2["predicate"]:
                lines[5],lines[9],lines[6],lines[10],lines[4],lines[8] = lines[9],lines[5],lines[10],lines[6],lines[8],lines[4]
                relation_dict["gene_others_physiology2"] += 1
                o.write("gene_others_physiology2," + ",".join(lines) + "\n")
            elif s_semtype in gene_others_physiology3["subject"] and o_semtype in gene_others_physiology3["object"] and not predicate in gene_others_physiology3["predicate"]:
                relation_dict["gene_others_physiology3"] += 1
                o.write("gene_others_physiology3," + ",".join(lines) + "\n")

            elif s_semtype in compound_others_gene1["subject"] and o_semtype in compound_others_gene1["object"] and not predicate in compound_others_gene1["predicate"]:
                relation_dict["compound_others_gene1"] += 1
                o.write("compound_others_gene1," + ",".join(lines) + "\n")
            elif s_semtype in compound_others_gene2["subject"] and o_semtype in compound_others_gene2["object"] and not predicate in compound_others_gene2["predicate"]:
                lines[5],lines[9],lines[6],lines[10],lines[4],lines[8] = lines[9],lines[5],lines[10],lines[6],lines[8],lines[4]
                relation_dict["compound_others_gene2"] += 1
                o.write("compound_others_gene2," + ",".join(lines) + "\n")
            elif s_semtype in compound_others_gene3["subject"] and o_semtype in compound_others_gene3["object"] and not predicate in compound_others_gene3["predicate"]:
                relation_dict["compound_others_gene3"] += 1
                o.write("compound_others_gene3," + ",".join(lines) + "\n")

            elif s_semtype in disease_others_disease1["subject"] and o_semtype in disease_others_disease1["object"] and not predicate in disease_others_disease1["predicate"]:
                relation_dict["disease_others_disease1"] += 1
                o.write("disease_others_disease1," + ",".join(lines) + "\n")
            elif s_semtype in disease_others_disease2["subject"] and o_semtype in disease_others_disease2["object"] and not predicate in disease_others_disease2["predicate"]:
                relation_dict["disease_others_disease2"] += 1
                o.write("disease_others_disease2," + ",".join(lines) + "\n")

            elif s_semtype in compound_others_disease1["subject"] and o_semtype in compound_others_disease1["object"] and not predicate in compound_others_disease1["predicate"]:
                relation_dict["compound_others_disease1"] += 1
                o.write("compound_others_disease1," + ",".join(lines) + "\n")

            elif s_semtype in bact_others_disease1["subject"] and o_semtype in bact_others_disease1["object"] and not predicate in bact_others_disease1["predicate"]:
                relation_dict["bact_others_disease1"] += 1
                o.write("bact_others_disease1," + ",".join(lines) + "\n")
            elif s_semtype in bact_others_disease2["subject"] and o_semtype in bact_others_disease2["object"] and not predicate in bact_others_disease2["predicate"]:
                lines[5],lines[9],lines[6],lines[10],lines[4],lines[8] = lines[9],lines[5],lines[10],lines[6],lines[8],lines[4]
                relation_dict["bact_others_disease2"] += 1
                o.write("bact_others_disease2," + ",".join(lines) + "\n")
            elif s_semtype in bact_others_disease3["subject"] and o_semtype in bact_others_disease3["object"] and not predicate in bact_others_disease3["predicate"]:
                relation_dict["bact_others_disease3"] += 1
                o.write("bact_others_disease3," + ",".join(lines) + "\n")
            elif s_semtype in bact_others_disease4["subject"] and o_semtype in bact_others_disease4["object"] and not predicate in bact_others_disease4["predicate"]:
                relation_dict["bact_others_disease4"] += 1
                o.write("bact_others_disease4," + ",".join(lines) + "\n")

            elif s_semtype in compound_others_compound["subject"] and o_semtype in compound_others_compound["object"] and not predicate in compound_others_compound["predicate"]:
                relation_dict["compound_others_compound1"] += 1
                o.write("compound_others_compound1," + ",".join(lines) + "\n")

            elif s_semtype in bact_others_compound1["subject"] and o_semtype in bact_others_compound1["object"] and not predicate in bact_others_compound1["predicate"]:
                relation_dict["bact_others_compound1"] += 1
                o.write("bact_others_compound1," + ",".join(lines) + "\n")
            elif s_semtype in bact_others_compound2["subject"] and o_semtype in bact_others_compound2["object"] and not predicate in bact_others_compound2["predicate"]:
                lines[5],lines[9],lines[6],lines[10],lines[4],lines[8] = lines[9],lines[5],lines[10],lines[6],lines[8],lines[4]
                relation_dict["bact_others_compound2"] += 1
                o.write("bact_others_compound2," + ",".join(lines) + "\n")
            elif s_semtype in bact_others_compound3["subject"] and o_semtype in bact_others_compound3["object"] and not predicate in bact_others_compound3["predicate"]:
                relation_dict["bact_others_compound3"] += 1
                o.write("bact_others_compound3," + ",".join(lines) + "\n")

            line = f.readline()
        for k,v in relation_dict.items():
            O.write("\t".join([k,str(v)])+"\n")