# -*- coding:utf-8 -*-
#This script use to generate train,validate,test dataset from positive and negative dataset

import os
import numpy as np
from random import shuffle
from sklearn.model_selection import train_test_split
from collections import defaultdict

if __name__ == "__main__":
    dataset = "./extension_predicate_class.csv.filtered_pmid.csv.selected_relation.csv.combine_text"
    negative_dataset = "./extension_predicate_class.csv.filtered_pmid.csv.negative_relation.csv.combine_text"
    relation = "./extension_predicate_class.csv.filtered_pmid.csv.selected_relation.csv.combine_text.relation.csv"
    train_dataset = dataset + ".train"
    test_dataset = dataset + ".test"
    val_dataset = dataset + ".val"
    dataset_info = dataset + ".spo_statics"
    relation_correct = relation + ".filtered_low_num"

    dataset_rel_dict = defaultdict(list)
    negative_dataset_rel_dict = defaultdict(list)
    #loading pos and neg dataset into list
    with open(dataset,"r") as f,open(negative_dataset,"r") as F:
        for line in f.readlines()[1:]:
            sents,predicate,s_name,s_idx,s_type,o_name,o_idx,o_type = line.rstrip("\n").rsplit(",",maxsplit=7)
            predicate = s_type.upper() + "_" + predicate.upper() + "_" + o_type.upper()
            dataset_rel_dict[(s_type,o_type,predicate)].append(",".join([sents,predicate,s_name,s_idx,o_name,o_idx])+"\n")
        for line in F.readlines()[1:]:
            sents,predicate,s_name,s_idx,s_type,o_name,o_idx,o_type = line.rstrip("\n").rsplit(",",maxsplit=7)
            predicate = s_type.upper() + "_" + predicate.upper() + "_" + o_type.upper()
            negative_dataset_rel_dict[(s_type,o_type,predicate)].append(",".join([sents,predicate,s_name,s_idx,o_name,o_idx])+"\n")
    with open(train_dataset,"w") as train_o,open(test_dataset,"w") as test_o,open(val_dataset,"w") as val_o,\
        open(dataset_info,"w") as static_o,open(relation_correct,"w") as rel_o:
        rel_o.write("head_type,tail_type,relation,index\nNone,None,None,0\n")
        train_o.write("sentence,relation,head,head_offset,tail,tail_offset\n")
        test_o.write("sentence,relation,head,head_offset,tail,tail_offset\n")
        val_o.write("sentence,relation,head,head_offset,tail,tail_offset\n")
        
        candidate_spo = {}
        idx = 1
        dataset_text_list = []
        dataset_target_list = []
        for k,v in dataset_rel_dict.items():
            #filter low num spo tripples type
            if len(v) >= 100:
                s_type,o_type,predicate = k
                predicate = s_type.upper() + "_OTHERS_" + o_type.upper()
                len_min_512_list = []
                for each_sample in v:
                    #bert max sequence length < 512
                    if len(each_sample) < 512:
                        len_min_512_list.append(each_sample)
                rand_positive_sample = []
                if len(v) >= 10000:
                    np.random.seed(42)
                    #generate random postive dataset
                    rand_positive_sample = np.random.choice(len_min_512_list,size=10000)
                else:
                    rand_positive_sample = len_min_512_list
                dataset_text_list.extend(rand_positive_sample)
                rel_o.write(",".join(k)+","+str(idx)+"\n")
                static_o.write(",".join(k) + "\t" + str(len(rand_positive_sample))+"\n")
                candidate_spo[(s_type,o_type,predicate)] = len(rand_positive_sample)
                idx += 1
                for _ in rand_positive_sample:
                    dataset_target_list.append(k)
        for k,v in negative_dataset_rel_dict.items():
            if len(v) >= 100 and k in candidate_spo:
                positive_sample_num = candidate_spo[k]
                np.random.seed(42)
                len_min_512_list = []
                for each_sample in v:
                    if len(each_sample) < 512:
                        len_min_512_list.append(each_sample)
                #generate random negative dataset
                rand_negative_sample = np.random.choice(len_min_512_list,size=positive_sample_num)
                dataset_text_list.extend(rand_negative_sample)
                rel_o.write(",".join(k)+","+str(idx)+"\n")
                static_o.write(",".join(k) + "\t" + str(len(rand_negative_sample))+"\n")
                idx += 1
                for _ in rand_negative_sample:
                    dataset_target_list.append(k)
            elif not k in candidate_spo:
                print(k,len(v))
        #train test stratify split,require train,validate,test dataset in same proportion
        train_data,nontrain_data,train_target,nontrain_target = train_test_split(dataset_text_list,dataset_target_list,test_size=0.1,\
            random_state=42,shuffle=True,stratify=dataset_target_list)
        val_data,test_data,val_target,test_target = train_test_split(nontrain_data,nontrain_target,test_size=0.4,random_state=42,shuffle=True,\
            stratify=nontrain_target)
        for train_d in train_data:
            train_o.write(train_d)
        for test_d in test_data:
            test_o.write(test_d)
        for val_d in val_data:
            val_o.write(val_d)
    target_path = "./semmedb_data/origin/"
    os.system("cp ./extension_predicate_class.csv.filtered_pmid.csv.selected_relation.csv.combine_text.t* %s"%target_path)
    os.system("cp ./extension_predicate_class.csv.filtered_pmid.csv.selected_relation.csv.combine_text.val %s"%target_path)
    os.system("cp ./extension_predicate_class.csv.filtered_pmid.csv.selected_relation.csv.combine_text.relation.csv.filtered_low_num %s"%target_path)
    os.system("mv %sextension_predicate_class.csv.filtered_pmid.csv.selected_relation.csv.combine_text.relation.csv.filtered_low_num %srelation.csv"%(target_path,target_path))
    os.system("mv %sextension_predicate_class.csv.filtered_pmid.csv.selected_relation.csv.combine_text.train %strain.csv"%(target_path,target_path))
    os.system("mv %sextension_predicate_class.csv.filtered_pmid.csv.selected_relation.csv.combine_text.test %stest.csv"%(target_path,target_path))
    os.system("mv %sextension_predicate_class.csv.filtered_pmid.csv.selected_relation.csv.combine_text.val %svalid.csv"%(target_path,target_path))

    

        