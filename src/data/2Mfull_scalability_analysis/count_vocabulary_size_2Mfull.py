
import pandas as pd


import csv

collect_vocabulary = set()
with open('/ceph/alebrink/tableAugmentation/data/2Mfull.csv', "r") as f:
    reader = csv.reader(f, delimiter="|")
    for i, line in enumerate(reader):
        if line[1] == 'Aggregate Value':
            continue
        for token in line[1].split(' '):
            collect_vocabulary.add(token)
        #if i == 10:
        #    break

        if i % 100000 == 0:
            print(i)

print(len(collect_vocabulary))

#df_2Mfull = pd.read_csv('/ceph/alebrink/tableAugmentation/data/2Mfull.csv', sep='|', quotechar='"', usecols=['Id', 'Aggregate Value', 'Clean Ag.Value'])

#print(df_2Mfull.head(5))