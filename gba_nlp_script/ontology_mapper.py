# -*- coding:utf-8 -*-

import sys
from collections import defaultdict
from unicodedata import name
import pandas as pd

def disease_xref_map(fname):
    disease_df = pd.read_csv(fname,sep="\t")
    return disease_df

def read_umls(fname):
    """Read through MRCONSO.RRF and extract relevant info.
    https://github.com/veleritas/merge_hetionet_semmeddb/blob/master/code/mappers/map_drugs.ipynb
    Currently extracted information:
        1. NCBI ID
        2. MESH

    Other data sources could be processed here, but diminishing returns kick
    in very quickly (they provide redundant data).

    For example, RxNorm mappings are almost a complete subset of the direct
    UNII mappings.

    Returns a pandas DataFrame with three columns.
    """
    res = defaultdict(list)
    with open(fname, "r") as fin:
        for line in fin:
            vals = line.rstrip("\n").split("|")

            cui, sab, code, name = vals[0], vals[11], vals[13], vals[14]

            if sab in {"NCBI","GO"}:
                res["cui"].append(cui)
                res["code"].append(code)
                res["source"].append(sab)
                res["name"].append(name)

    return (pd
        .DataFrame(res)
        .drop_duplicates()
        .reset_index(drop=True)
    )

def ontology_map(entity_type,entity_id):
    if entity_type == "gene":
        entity_id = [entity_id.split(":")[1]]
    elif entity_type == "disease":
        new_entity_id = ""
        disease_id = entity_id.split(":")[1]
        if "omim" in entity_id:
            try:
                new_entity_id = disease_mapper[(disease_mapper.resource_id==disease_id)\
                    &(disease_mapper.resource=="OMIM")].doid_code.values
            except:
                pass
        elif "mesh" in entity_id:
            try:
                new_entity_id = disease_mapper[(disease_mapper.resource_id==disease_id)\
                    &(disease_mapper.resource=="MESH")].doid_code.values
            except:
                pass
        entity_id = new_entity_id if len(new_entity_id)!=0 else [entity_id]
    elif entity_type == "compound":
        entity_id = [entity_id]
    elif entity_type == "bact":
        try:
            entity_id = umls_mapper[(umls_mapper.cui==entity_id)&(umls_mapper.source=="NCBI")]\
                .code.values
        except:
            print(entity_id,entity_type)
    elif entity_type == "physiology":
        try:
            entity_id = umls_mapper[(umls_mapper.cui==entity_id)&(umls_mapper.source=="GO")]\
                .code.values
        except:
            print(entity_id,entity_type)
            entity_id = ""

    return entity_type,entity_id
if __name__ == "__main__":
    MRCONSO_file = "/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/MRCONSO.RRF"
    disease_xref = "/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/HumanDO.owl.map"

    ontology_class_set = {"gene","compound","disease","bact","physiology"}

    ner_re_output_csv = sys.argv[1]
    ner_re_output_csv_unified = ner_re_output_csv + ".entity_aligned"

    umls_mapper = read_umls(MRCONSO_file)
    disease_mapper = disease_xref_map(disease_xref)

    ner_re_sents_dict = defaultdict(list)
    with open(ner_re_output_csv,"r") as f,open(ner_re_output_csv_unified,"w") as o:
        for line in f.readlines():
            lines = line.rstrip().split("\t")
            sents,relationship = lines
            text,s_prefer_name,s_origin_name,s_type,s_start,s_id,o_prefer_name,o_origin_name,\
                o_type,o_start,o_id = sents.rsplit("|",maxsplit=10)
            if s_type and o_type in ontology_class_set:
                s_type,s_id = ontology_map(s_type,s_id)
                o_type,o_id = ontology_map(o_type,o_id)
                for single_s_id in s_id:
                    for single_o_id in o_id:
                        ner_re_sents_dict[text].append("|".join([s_prefer_name,s_type,single_s_id,\
                            relationship,o_prefer_name,o_type,single_o_id]))
            else:
                continue
        for idx,kvs in enumerate(ner_re_sents_dict.items()):
            text,vs = kvs
            for each_kv in vs:
                o.write("\t".join([str(idx+1),each_kv])+"\n")
