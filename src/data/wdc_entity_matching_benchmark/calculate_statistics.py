
import pandas as pd
# Load file into pandas dataframe
df = pd.read_csv('data/deepmatcher/wdcproducts80cc20rnd050un_block_l_train_l/all.csv', sep=',', encoding='utf-8')

# # Print length of dataframe
# print('Length of dataframe: {}'.format(len(df)))
#
# # Assign records to clusters
# clusters = {}
# for index, row in df.iterrows():
#     if row['label'] == 0:
#         continue
#     if row['cluster_id_left'] not in clusters:
#         clusters[row['cluster_id_left']] = set()
#     clusters[row['cluster_id_left']].add(row['ltable_id'])
#
#     if row['cluster_id_right'] not in clusters:
#         clusters[row['cluster_id_right']] = set()
#     clusters[row['cluster_id_right']].add(row['rtable_id'])
#
# # Print number of clusters
# print('Number of clusters: {}'.format(len(clusters)))
#
# # Print largest cluster
# largest_cluster = 0
# content_largest_cluster = set()
# largest_cluster_id = ''
# for cluster_id in clusters:
#     if len(clusters[cluster_id]) > largest_cluster:
#         largest_cluster = len(clusters[cluster_id])
#         content_largest_cluster = clusters[cluster_id]
#         largest_cluster_id = cluster_id
# print('Largest cluster: {}'.format(largest_cluster))
# print('Content largest cluster: {}'.format(content_largest_cluster))
# print('Largest cluster ID: {}'.format(largest_cluster_id))

df_tableA = pd.read_csv('data/deepmatcher/wdcproducts80cc20rnd050un_block_l_train_l/tableA.csv', sep=',', encoding='utf-8')
df_tableB = pd.read_csv('data/deepmatcher/wdcproducts80cc20rnd050un_block_l_train_l/tableB.csv', sep=',', encoding='utf-8')

# Count cluster sizes
cluster_sizes = {}
for index, row in df_tableA.iterrows():
    if row['cluster_id'] not in cluster_sizes:
        cluster_sizes[row['cluster_id']] = 0
    cluster_sizes[row['cluster_id']] += 1

for index, row in df_tableB.iterrows():
    if row['cluster_id'] not in cluster_sizes:
        cluster_sizes[row['cluster_id']] = 0
    cluster_sizes[row['cluster_id']] += 1

# Determine largest cluster
largest_cluster = 0
largest_cluster_id = ''
for cluster_id in cluster_sizes:
    if cluster_sizes[cluster_id] > largest_cluster:
        largest_cluster = cluster_sizes[cluster_id]
        largest_cluster_id = cluster_id

print('Largest cluster: {}'.format(largest_cluster))
print('Largest cluster ID: {}'.format(largest_cluster_id))
