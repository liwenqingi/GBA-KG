# -*- coding:utf-8 -*-
#genernating semmedb data from PREDICATION.csv and SENTENCE.csv
#semmedb data url:https://lhncbc.nlm.nih.gov/ii/tools/SemRep_SemMedDB_SKR/SemMedDB_download.html
#require file
#1.PREDICATION.csv url:/share/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/3_mysql_load/PREDICATION.csv
#2.SENTENCE.csv url:/share/home/liwenqing/liwenqing_hpc/5_neo4j_3.5/4_deepke_paper/3_mysql_load/SENTENCE.csv

import sys
import os

path = "./semmedb_loading_script"

os.system("python %s/extract_extensionclass_from_semmedb.py"%path)
os.system("python %s/extract_common_pmid_from_extensionclass.py"%path)
os.system("python %s/extract_relationship_from_filtered_csv.py ./extension_predicate_class.csv.filtered_pmid.csv"%path)
os.system("python %s/extract_negative_relationship_from_filtered_csv.py ./extension_predicate_class.csv.filtered_pmid.csv"%path)
os.system("python %s/generate_positive_dataset_from_selected_csv.py"%path)
os.system("python %s/generate_negative_dataset_from_extension_csv.py"%path)
os.system("python %s/generate_train_val_test_data.py"%path)