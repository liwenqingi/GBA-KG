# GBA-KG 

![Release](https://img.shields.io/badge/Release-Ver1.0.0-blue.svg)

Gut-brain axis knowledge graph(GBA-KG) is a knowledge graph that integrates existingknowledge graphs in the biomedical field and information on microbiology and psychiatry in the literature. GBA-KG integrates a large number of biomedical resources, describing the relationship between microorganisms, genes, diseases, anatomy, compounds, molecular functions, biological processes, cellular components and pathways, expanding the knowledge map in microbial psychiatric diseases. We describe the process from prototyping GBA-KG to information extraction and information updating. It is convenient for researchers to learn, and multi-modal analysis can be combined with machine learning algorithms in the future.  

## Cloning and installing 

The setting up of the GBA is easy and might just take a few mintues.To get a copy of the GitHub repository on your local machine, please open a terminal windown and run:  
```
git clone https://github.com/liwenqingi/GBA-KG.git
```  
This will create a new folder named "GBA-KG" on your current location. After this, follow the instructions in "Building GBA-KG". If you want to use following NLP tools, check the `requirements.txt` for detail. 
## Building GBA-KG  
The construction of KG consists of two main steps, the first is to integrate multiple resources, the second is to acquire knowledge from literature and update it. Before building KGs, you should start neo4j docker(build via `Dockerfile`) and bern2 firstly following `run_bern2.sh` and `demo_graphdb_owl_neo4j_3.5.sh`. 
### Integrate multiple resources  
Ontology resources can be load into neo4j by `run_kg_creation.py`. After running this script, you should get results like `neo4j.png` via web browser. 

The ontology databases and hetionet knowledge graph provided within the GBA-KG have their own licenses and the use of GBA-KG still requires compliance with these data use restrictions. Please, visit the data sources directly for more information:  

| Source type | Source | URL |
| --- | --- | --- |
| Ontology | Human Disease Ontology | https://www.ebi.ac.uk/ols/ontologies/doid | 
| Ontology | OGG: Ontology of Genes and Genomes | https://www.ebi.ac.uk/ols/ontologies/ogg |
| Ontology | NCBI organismal classification | https://obofoundry.org/ontology/ncbitaxon.html |
| Ontology | Chemical Entities of Biological Interest | https://www.ebi.ac.uk/ols/ontologies/chebi |
| Ontology | Uber-anatomy ontology | https://www.ebi.ac.uk/ols/ontologies/uberon |
| Ontology | Gene Ontology | https://www.ebi.ac.uk/ols/ontologies/go |
| Knowledge Graph | Hetionet | https://github.com/hetio/hetionet | 
### Literature knowledge acquisition
Literature information can be manually organized into triples and imported. For the currently supported relationships, please refer to the `semmedb_data/origin/relation.csv` file. At the same time, we use NLP methods to accelerate this process.  
![Process](https://github.com/liwenqingi/GBA-KG/blob/main/NLP_process.png) 
#### 1.Choose study field
By specifying field keywords in the `generate_kg_pubmed_abstracts.py` script, the corresponding literature abstracts can be downloaded automatically. Examples can refer to `kg_pubmed_abstracts.csv`.  
#### 2.Generate training data
The data used in training comes from [SemMedDB](https://lhncbc.nlm.nih.gov/ii/tools/SemRep_SemMedDB_SKR/SemMedDB_download.html), triples are extracted from `PREDICATION`, and sentence information of triples is extracted from `SENTENCE`. We provide a processed dataset `semmedb_data/origin`, you can skip directly to the data generation stage if study field matches.
#### 3.Information extraction
Information extraction contains NER and RE,there is a packed script `run_nlp_kg.py` that can be used directly.Named entity recognition using [pymetamap](https://github.com/AnthonyMRios/pymetamap) and [BERN2](https://github.com/dmis-lab/BERN2). pymetamap is used to identify microbes, anatomy, BERN2 is used to identify other entity concepts.The module of relation extraction refers to [DeepKE](https://github.com/zjunlp/DeepKE) and the core algorithm is biobert+BiLSTM.We provide the fine-tuned [biobert model](https://drive.google.com/drive/u/0/my-drive) so you can skip model-training stage. Examples can refer to `kg_pubmed_abstracts.csv.ner.predict`.
#### 4.Cleaning and entity alignment
Remove entities with unrecognized id and duplicate entities. Match the identified entity id with the id of the corresponding entity library.Examples can refer to `kg_pubmed_abstracts.csv.ner.predict.entity_aligned`.
#### 5.Data loading
Import data into neo4j database.  
## Acknowledgments
* [Hetionet](https://github.com/hetio/hetionet)
* [DeepKE](https://github.com/zjunlp/DeepKE)
* [SemMedDB](https://lhncbc.nlm.nih.gov/ii/tools/SemRep_SemMedDB_SKR/SemMedDB_download.html)
## Contact
If you have any constructive comments or other ideas, please contact me directly on github or send email to liwenqingi@163.com.
