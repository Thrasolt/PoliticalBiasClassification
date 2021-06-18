from typing import List

from SPARQLWrapper import SPARQLWrapper, JSON

from src.main.python.data_retrieval import prefixes

sparql = SPARQLWrapper("https://data.gesis.org/tweetscov19/sparql")
sparql.setReturnFormat(JSON)


def extract_results_from_query(key: str) -> List:
    results = sparql.query().convert()
    values = [result[key]["value"] for result in results["results"]["bindings"]]
    return values


def get_hashtags_for_pld(pld: str) -> List[str]:
    sparql.setQuery(f"""
        {prefixes.header}

        SELECT ?hashtag
        WHERE {{
         ?tweet schema:citation ?url.

        ?tweet schema:mentions ?tag.
        ?tag rdfs:label ?hashtag.

         filter( regex( str(?url),"{pld}" ) )

        }}
    """)
    return extract_results_from_query("hashtag")


def get_emotion_for_pld(pld: str, category: str) -> List[str]:
    sparql.setQuery(f"""
        {prefixes.header}

        SELECT ?emo_int
        WHERE {{
         ?tweet schema:citation ?url.

         ?tweet onyx:hasEmotionSet ?emotion_set.
         ?emotion_set onyx:hasEmotion ?emotion.
         ?emotion onyx:hasEmotionIntensity ?emo_int.
         ?emotion onyx:hasEmotionCategory wna:{category}-emotion.
         filter( regex( str(?url),"{pld}" ) )
        }}
    """)
    return extract_results_from_query("emo_int")


def get_mentioned_entities_for_pld(pld: str) -> List[str]:
    sparql.setQuery(f"""
        {prefixes.header}

        SELECT ?mentioned_entity
        WHERE {{
         ?tweet schema:citation ?url.

         ?tweet schema:mentions ?entity.
         ?entity nee:detectedAs ?mentioned_entity.
         
         filter( regex( str(?url),"{pld}" ) )
        }}
    """)
    return extract_results_from_query("mentioned_entity")


def get_mentioned_accounts_for_pld(pld: str) -> List[str]:
    sparql.setQuery(f"""
        {prefixes.header}

        SELECT ?mentioned_account
        WHERE {{
         ?tweet schema:citation ?url.

         ?tweet schema:mentions ?account.
         ?account sioc:name ?mentioned_account.

         filter( regex( str(?url),"{pld}" ) )
        }}
    """)
    return extract_results_from_query("mentioned_account")


def retrieve_data_for_pld(pld: str):
    return [
        pld,
        get_hashtags_for_pld(pld),
        get_emotion_for_pld(pld, "positive"),
        get_emotion_for_pld(pld, "negative"),
        get_mentioned_entities_for_pld(pld),
        get_mentioned_accounts_for_pld(pld)
    ]