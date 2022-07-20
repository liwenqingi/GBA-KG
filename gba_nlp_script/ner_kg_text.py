# -*- coding:utf-8 -*-

import sys
import time
import requests
import pymetamap
from interval import Interval
from collections import defaultdict

#loading umls semantic for subsequent entity determination
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

#query bern2 json
def query_plain(text, url="http://localhost:8888/plain"):
    return requests.post(url, json={'text': text}).json()

#specifies umls entity that metamap need to query 
def pymetamap_candidate_settings():
    total_use_group = set(("Anatomy","Physiology"))
    part_use_group = {"Living Beings":set(["bact"])}
    semantic_dict = load_umls_semantic_dict()
    candidate_class = []
    for k in semantic_dict:
        semantic_group = semantic_dict[k]["group"]
        if semantic_group in total_use_group:
            candidate_class.append(k)
        elif semantic_group in part_use_group:
            if k in part_use_group[semantic_group]:
                candidate_class.append(k)
    candidate_class = ",".join(candidate_class)
    #candidate_class = "bpoc,tisu,bact,gngm,antb,dsyn,"
    return candidate_class
    
#load RE relation.csv for subsequent relationship determination
def load_relation_csv(rel_path):
    rel_dict = defaultdict(set)
    with open(rel_path, "r") as f:
        for line in f.readlines()[2:]:
            s,o,p,_ = line.rstrip("\n").split(",")
            rel_dict[s].add(o)
    return rel_dict


def main():
    #text format like /home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/2_test_biobert/test_abstract.txt
    txt_file = sys.argv[1]
    #text format like /home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/2_test_biobert/test_abstract.txt/test_abstract.txt.output
    output_file = sys.argv[2]
    relation_path = "/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/2_test_biobert/data/origin/relation.csv"
    #map bern2 class name to neo4j class name
    bern2_ncbi_class_mapper = {"drug":"compound","gene":"gene","DNA":"gene","RNA":"gene","disease":"disease"}
    #map metamap class name to neo4j class name
    pymetamap_class_mapper = {"Anatomy":"anatomy","Living Beings":"bact","Physiology":"physiology"}
    #loading relation csv
    relation_dict = load_relation_csv(relation_path)
    #loading pymetamap csv
    pymetamap_pred_class = pymetamap_candidate_settings()
    
    metamap_path = "/home/liwenqing/liwenqing_hpc/2_software/MetaMap/public_mm/bin/metamap20"    
    mm = pymetamap.MetaMap.get_instance(metamap_path)

    semantic_dict = load_umls_semantic_dict()
    #reading input text file
    with open(txt_file,"r") as f:
        txt_list = []
        for line in f.readlines():
            try:
                txt_id,txt_title,txt_abs = line.rstrip("\n").split("|",maxsplit=2)
                txt_list.append(txt_abs)
            except:
                print(line)            
        txt_list = [x.replace("\n"," ") for x in txt_list]
    #write output csv
    with open(output_file,"w") as o:
        for idx,texts in enumerate(txt_list):
            #split total abstract
            text_list = texts.split(".")
            for text in text_list:
                if text:
                    text = text.strip()
                    entity_candidate_list = []
                    #query bern2 json
                    bern2_json = query_plain(text)
                    #query pymetamap concept
                    concepts,error = mm.extract_concepts([text],restrict_to_sts=[pymetamap_pred_class],prune=4)
                    for each_spo in bern2_json["annotations"]:
                        #The entity must have a unique ID
                        if each_spo["obj"] in bern2_ncbi_class_mapper and each_spo["id"][0]!="CUI-less":
                            each_spo["semtype"],each_spo["source"] = bern2_ncbi_class_mapper[each_spo["obj"]],"bern2"
                            each_spo["begin"],each_spo["end"] = each_spo["span"]["begin"], each_spo["span"]["end"]
                            #bern2 preferred_name is mention in dict
                            each_spo["preferred_name"] = each_spo["mention"]
                            entity_candidate_list.append(each_spo)
                    pos_dict = defaultdict(dict)
                    for concept in concepts:
                        try:
                            #concept object must be conceptMM
                            name,semtype,cui,pos_info,score = concept.preferred_name,concept.semtypes,concept.cui,concept.pos_info,\
                                float(concept.score)
                        except:
                            continue
                        #Requires no Spaces between a single entity
                        if not cui == "CUI-less" and not "," in pos_info:
                            semtype = semtype.replace("[","").replace("]","").split(",")[0]
                            #require entity type in umls
                            semtype = pymetamap_class_mapper[semantic_dict[semtype]["group"]]
                            #parse position info
                            for each_pos,each_trigger in zip(pos_info.split(";"),concept.trigger.split(",")):
                                start_pos,span = each_pos.split("/")
                                mention = each_trigger.split("\"")[-2]
                                #1-based - 1
                                start_pos = int(start_pos)-1
                                end_pos = start_pos + int(span)
                                concept_dict = {"source":"pymetamap","begin":start_pos,"end":end_pos,"preferred_name":name,"cui":cui,\
                                    "semtype":semtype,"mention":mention,"id":[cui]}
                                #There may be multiple entities in the same position, and the one with the highest score is selected
                                if not each_pos in pos_dict:
                                    pos_dict[each_pos]["data"] = concept_dict
                                    pos_dict[each_pos]["score"] = score
                                elif score > pos_dict[each_pos]["score"]:
                                    pos_dict[each_pos]["data"] = concept_dict
                                    pos_dict[each_pos]["score"] = score
                    for k,v in pos_dict.items():
                        entity_candidate_list.append(v["data"])
                    #Iterate through the list of entities, picking out those whose positions do not overlap
                    for pre_spo_idx in range(len(entity_candidate_list)-1):
                        pre_spo_entity = entity_candidate_list[pre_spo_idx]
                        pre_start,pre_end,pre_semtype = pre_spo_entity["begin"],pre_spo_entity["end"],pre_spo_entity["semtype"]
                        pre_spo_mention,pre_id = pre_spo_entity["mention"],pre_spo_entity["id"][0]
                        pre_spo_name = pre_spo_entity["preferred_name"]
                        pre_spo_span = Interval(pre_start,pre_end,lower_closed=True,upper_closed=False)
                        for tail_spo_idx in range(pre_spo_idx+1,len(entity_candidate_list)):
                            tail_spo_entity = entity_candidate_list[tail_spo_idx]
                            tail_start,tail_end,tail_semtype = tail_spo_entity["begin"],tail_spo_entity["end"],tail_spo_entity["semtype"]
                            tail_spo_span = Interval(tail_start,tail_end,lower_closed=True,upper_closed=False)
                            tail_spo_mention,tail_id = tail_spo_entity["mention"],tail_spo_entity["id"][0]
                            tail_spo_name = tail_spo_entity["preferred_name"]
                            if not pre_spo_span.overlaps(tail_spo_span) and not pre_id == tail_id:
                                if pre_semtype in relation_dict and tail_semtype in relation_dict[pre_semtype]:
                                    o.write(str(idx+1)+"|"+"|".join([text,pre_spo_name,pre_spo_mention,pre_semtype,str(pre_start),pre_id,tail_spo_name,\
                                        tail_spo_mention,tail_semtype,str(tail_start),tail_id])+"\n")
                                elif tail_semtype in relation_dict and pre_semtype in relation_dict[tail_semtype]:
                                    o.write(str(idx+1)+"|"+"|".join([text,tail_spo_name,tail_spo_mention,tail_semtype,str(tail_start),tail_id,pre_spo_name,\
                                        pre_spo_mention,pre_semtype,str(pre_start),pre_id])+"\n")
if __name__ == '__main__':
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()
    print("time used:%.1fs"%(end_time-start_time))
        

    
        