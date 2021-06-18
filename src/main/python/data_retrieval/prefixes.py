nee: str = "nee: <http://www.ics.forth.gr/isl/oae/core#>"
sioc: str = "sioc: <http://rdfs.org/sioc/ns#>"
schema: str = "schema: <http://schema.org/>"
dc: str = "dc: <http://purl.org/dc/terms/>"
dbr: str = "dbr: <http://dbpedia.org/resource/>"
rdfs: str = "rdfs: <http://www.w3.org/2000/01/rdf-schema#>"
onyx: str = "onyx: <http://www.gsi.dit.upm.es/ontologies/onyx/ns#>"
wna: str = "wna: <http://www.gsi.dit.upm.es/ontologies/wnaffect/ns#>"

header: str = f"""
    PREFIX {nee}
    PREFIX {sioc}
    PREFIX {schema}
    PREFIX {dc}
    PREFIX {dbr}
    PREFIX {rdfs}
    PREFIX {onyx}
    PREFIX {wna}
"""
