"""
use to extract pumed abstracts,needs to specify the key words in asd_forum_url function
https://erilu.github.io/pubmed-abstract-compiler/
https://github.com/chengjiali/pubmed_corpus/blob/master/fetch_pmid_list.py
"""
from ast import BinOp
import os
import csv
import sys
import json
import re
import urllib
import requests
from time import sleep

#key words:mitochondria+neurodevelopment
def asd_form_url():
    return f'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&\
term=mitochondria+neurodevelopment&usehistory=y&rettype=json'

if __name__ == "__main__":
    if os.path.exists("./kg_pubmed_abstracts.csv"):
        os.remove("./kg_pubmed_abstracts.csv")

    search_url = asd_form_url()
    
    f = urllib.request.urlopen(search_url)
    search_data = f.read().decode('utf-8')
    # extract the total abstract count
    total_abstract_count = int(re.findall("<Count>(\d+?)</Count>",search_data)[0])
    print("total abstracts:",total_abstract_count)
    # efetch settings
    fetch_eutil = 'efetch.fcgi?'
    retmax = 100
    retstart = 0
    fetch_retmode = "&retmode=text"
    fetch_rettype = "&rettype=abstract"
    # obtain webenv and querykey settings from the esearch results
    fetch_webenv = "&WebEnv=" + re.findall ("<WebEnv>(\S+)<\/WebEnv>", search_data)[0]
    fetch_querykey = "&query_key=" + re.findall("<QueryKey>(\d+?)</QueryKey>",search_data)[0]

    base_url = 'http://eutils.ncbi.nlm.nih.gov/entrez/eutils/'
    db = 'db=pubmed'
    # call efetch commands using a loop until all abstracts are obtained
    run = True
    all_abstracts = list()
    loop_counter = 1
    c = 0
    #output file
    abstracts_csv = open("./kg_pubmed_abstracts.csv","a")
    #abstracts_writer = csv.writer(abstracts_csv)
    #abstracts_writer.writerow(['Journal', 'Title', 'Authors', 'Author_Information', 'Abstract', 'DOI', 'Misc'])
    while run:
        print("this is efetch run number " + str(loop_counter))
        loop_counter += 1
        fetch_retstart = "&retstart=" + str(retstart)
        fetch_retmax = "&retmax=" + str(retmax)
        # create the efetch url
        fetch_url = base_url+fetch_eutil+db+fetch_querykey+fetch_webenv+fetch_retstart+\
            fetch_retmax+fetch_retmode+fetch_rettype
        # open the efetch url
        f = urllib.request.urlopen (fetch_url)
        fetch_data = f.read().decode('utf-8')
        # split the data into individual abstracts
        abstracts = fetch_data.split("\n\n\n")
        for simple_abstract in abstracts:
            abstract_split = simple_abstract.split("\n\n")
            #len(abstracts) < 5 does not have total abstracts
            if len(abstract_split) == 5:
                abstracts_id,abstracts_title,abstracts_text = abstract_split[-1],abstract_split[1],abstract_split[-2]
            elif len(abstract_split) == 6:
                abstracts_id,abstracts_title,abstracts_text = abstract_split[-1],abstract_split[1],abstract_split[-2]
            elif len(abstract_split) == 7:
                abstracts_id,abstracts_title,abstracts_text = abstract_split[-1],abstract_split[1],abstract_split[-3]
            elif len(abstract_split) == 8:
                abstracts_id,abstracts_title,abstracts_text = abstract_split[-1],abstract_split[1],abstract_split[-4]
            elif len(abstract_split) == 9:
                abstracts_id,abstracts_title,abstracts_text = abstract_split[-1],abstract_split[1],abstract_split[-5]            
            abstracts_id = abstracts_id.replace("\n"," ")
            abstracts_title = abstracts_title.replace("\n"," ")
            abstracts_text = abstracts_text.replace("\n"," ")
            abstracts_csv.write(abstracts_id+"|"+abstracts_title+"|"+abstracts_text+"\n")
            c += 1
        print("a total of " + str(c) + " abstracts have been downloaded.\n")
        # wait 0.1 seconds so we don't get blocked
        sleep(0.1)
        # update retstart to download the next chunk of abstracts
        retstart = retstart + retmax
        if retstart > total_abstract_count:
            run = False
    abstracts_csv.close()
