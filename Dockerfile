FROM neo4j:3.5
#docker run -it --rm -p 7474:7474 -p 7687:7687 --entrypoint=bash neo4j:3.4
#WORKDIR /var/lib/neo4j/data/databases/
#RUN rm -rf graph.db
#RUN wget -c -t 100 https://github.com/hetio/hetionet/raw/master/hetnet/neo4j/hetionet-v1.0.db.tar.bz2
#COPY graph.db/ graph.db/
WORKDIR /var/lib/neo4j/conf/
COPY neo4j.conf neo4j.conf
WORKDIR /var/lib/neo4j/plugins
COPY neosemantics-3.5.0.4.jar neosemantics-3.5.0.4.jar
WORKDIR /var/lib/neo4j/bin
ENTRYPOINT ["neo4j","console"]
