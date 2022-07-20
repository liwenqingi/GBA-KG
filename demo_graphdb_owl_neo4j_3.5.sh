#!/usr/bin/bash
#mount_abs_file=$1
#mount_file=`basename $1`

docker run --user $(id -u):$(id -g) --rm -p 7474:7474 -p 7687:7687 --env NEO4J_AUTH=neo4j/123 -v /home/liwenqing/liwenqing_hpc/5_neo4j_3.5/3_demo2_ncbitaxon_owl_neo4j_3.5/graph.db:/var/lib/neo4j/data/databases/graph.db  -v  /home/liwenqing/liwenqing_hpc/5_neo4j_3.5/neo4j.conf:/var/lib/neo4j/conf/neo4j.conf  neo4j_3.5
