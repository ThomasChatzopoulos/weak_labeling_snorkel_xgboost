import xmltodict
import json
import csv
import random
import os
import pandas as pd
from datetime import datetime

path = "C:/Users/Documents/files"


def fileName(n):
    filename = ""
    if len(str(n)) == 1:
        filename = f"pubmed21n000{n}.xml"
    elif len(str(n)) == 2:
        filename = f"pubmed21n00{n}.xml"
    elif len(str(n)) == 3:
        filename = f"pubmed21n0{n}.xml"
    elif len(str(n)) == 4:
        filename = f"pubmed21n{n}.xml"
    return filename


# Load pmids of WS gains negative list from error_analysis_report.csv
pmids = []
report = pd.read_csv(f"{path}/error_analysis.csv")
for articles in report["WS gains negative list"]:
    if len(articles) > 2:
        pmids += articles.replace("[", "").replace("]", "").replace(",", "").split(" ")

pmidset = set(pmids)
print(f"len of pmids: {len(pmids)}\nlen of unique: {len(pmidset)}\n")

# # search for pmids into pubmed files-baseline
# json_file = open(f'{path}/articleSet.json', "a")
# json_file.write('{"articleSet":{')
# json_file.close()
# for i in range(1015, 0, -1):
# # for i in range(1113, 1115):
#     filename = fileName(i)
#     print(f'{datetime.now().strftime("%d/%m/%Y_%H:%M:%S")}_{filename}')
#     filestream = open(f'{path}/{filename}', encoding='utf-8')
#     xml_dictionary = xmltodict.parse(filestream.read(), encoding='utf-8')
#     filestream.close()
#     for article in xml_dictionary.get('PubmedArticleSet').get('PubmedArticle'):
#         article_pmid = article.get('MedlineCitation').get('PMID').get('#text')
#         articleSet = {}
#         if article_pmid in pmids:
#             json_file = open(f'{path}/articleSet.json', "a")
#             id = f'{article_pmid}_{i}_{datetime.now().strftime("%S")}{random.randint(100, 999)}'
#             json_file.write(f'\n"{id}":')
#             # articleSet[f'{article_pmid}_{i}_{datetime.now().strftime("%S")}{random.randint(100, 999)}'] = article
#             json.dump(article, json_file)
#             json_file.write(',')
#             json_file.close()
# # remove last comma
# with open(f'{path}/articleSet.json', 'rb+') as filehandle:
#     filehandle.seek(-1, os.SEEK_END)
#     filehandle.truncate()
# json_file = open(f'{path}/articleSet.json', "a")
# json_file.write('}}')
# json_file.close()
#

# create summary file
pmidsfound = []
jsonfilestream = open(f"{path}/articleSet.json", "r")
with open(f"{path}/results.csv", "w", encoding="UTF8", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["PMID", "part", "indexingMethod"])
    articles = json.load(jsonfilestream)
    for article in articles["articleSet"]:
        articlePMID = article.split("_")[0]
        pmidsfound.append(articlePMID)
        part = article.split("_")[1]
        indexingMethod = "null"
        if "@IndexingMethod" in articles["articleSet"].get(article).get(
            "MedlineCitation"
        ):
            indexingMethod = (
                articles["articleSet"]
                .get(article)
                .get("MedlineCitation")
                .get("@IndexingMethod")
            )
        writer.writerow([articlePMID, part, indexingMethod])
jsonfilestream.close()

pmidfoundset = set(pmidsfound)
print(f"len of pmidsfound: {len(pmidsfound)}\nlen of unique: {len(pmidfoundset)}\n")

out = []
for id in pmidsfound:
    if id not in pmids:
        out.append(id)

print(f"out: {out}")

seen = set()
dupes = []
for x in pmids:
    if x in seen:
        dupes.append(x)
    else:
        seen.add(x)

print(f"dup: {dupes}")
