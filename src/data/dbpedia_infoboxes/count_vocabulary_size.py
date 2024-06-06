
import os
from urllib.parse import unquote

# Read goldstandard entities
goldstandard_entities = set()
with open('/ceph/alebrink/tableAugmentation/data/large_scale_benchmarks/BTC12DBPedia_Infoboxes/dbpedia2dbpediaGroundTruth.txt', "r") as f:
    for line in f:
        goldstandard_entities.add(unquote(line.strip()).replace('http://dbpedia.org/resource/', ''))

print(len(goldstandard_entities))
print(sorted(list(goldstandard_entities))[:5])

class2id_dbpedia = dict()

overlap_count = 0
overlapping_entities = set()

print('Reading DBpedia infoboxes...')
infobox2ids = {}
id2infobox = {}
with open('/ceph/alebrink/tableAugmentation/data/large_scale_benchmarks/BTC12DBPedia_Infoboxes/infoboxIds.txt', "r") as f:
    for line in f:
        values = line.replace('\n','').split('\t')
        class_value = unquote(values[0].replace('<','').replace('>','').replace('dbp:', '').strip())
        infobox2ids[class_value] = values[1].strip()
        id2infobox[values[1].strip()] = class_value

print('Reading DBpedia...')
dbpedia2ids = {}
id2dbpedia = {}
# Read record IDs from BTC12DBPedia
with open('/ceph/alebrink/tableAugmentation/data/large_scale_benchmarks/BTC12DBPedia_Infoboxes/dbpediaIds.txt', "r") as f:
    for line in f:
        values = line.replace('\n','').split('\t')
        class_value = unquote(values[0].replace('<','').replace('>','').replace('dbp:', '').strip())
        dbpedia2ids[class_value] = values[1].strip()
        id2dbpedia[values[1].strip()] = class_value

        if class_value in infobox2ids:
            overlap_count += 1
            overlapping_entities.add(class_value)

print('Overlap count(unique): {}'.format(len(overlapping_entities)))
print('Overlap count: {}'.format(overlap_count))
print('Total DBpedia entities: {}'.format(len(dbpedia2ids)))
print('Total DBpedia infoboxes: {}'.format(len(infobox2ids)))
# Determine vocabulary size
collect_vocabulary = set()

found_ids = set()
print('Reading DBpedia infoboxes...')
with open('/ceph/alebrink/tableAugmentation/data/large_scale_benchmarks/BTC12DBPedia_Infoboxes/infoboxEntityIds.nt', "r") as f:
    current_id = ''
    for line in f:
        values = line.replace('\n','').split(' ')
        if current_id != values[0]:
            current_id = values[0]
            if current_id in id2infobox:
                found_ids.add(current_id)
                collect_vocabulary.add(values[1].replace('<prop:','').replace('>',''))
                for value in values[2:]:
                    collect_vocabulary.add(value.replace('<prop:','').replace('>',''))

print('Reading DBpedia...')
with open('/ceph/alebrink/tableAugmentation/data/large_scale_benchmarks/BTC12DBPedia_Infoboxes/dbpedia37EntityIds.nt', "r") as f:
    current_id = ''
    for line in f:
        values = line.replace('\n','').split(' ')
        if current_id != values[0]:
            current_id = values[0]
            if current_id in id2dbpedia:
                found_ids.add(current_id)
                collect_vocabulary.add(values[1].replace('<prop:','').replace('>',''))
                for value in values[2:]:
                    collect_vocabulary.add(value.replace('<prop:','').replace('>',''))

print('Vocabulary size: {}'.format(len(collect_vocabulary)))
print('Found IDs: {}'.format(len(found_ids)))
