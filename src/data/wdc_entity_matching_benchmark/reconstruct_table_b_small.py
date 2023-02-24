import pandas as pd

# Load table from CSV file
table_csv = pd.read_csv('C:/Users/alebrink/Documents/02_Research/TableAugmentation/table-augmentation-framework/data/deepmatcher/wdcproducts80cc20rnd050un/tableB.csv')

print(f'Table from CSV file: \n{table_csv}')

# Load table from JSON file
table_json = pd.read_json('C:/Users/alebrink/Documents/02_Research/TableAugmentation/table-augmentation-framework/data/corpus/wdcproducts80cc20rnd050un/wdcproducts80cc20rnd050un_tableB.json/wdcproducts80cc20rnd050un_tableB.json', lines=True)

print(table_csv.columns)
print(table_json.columns)
filtered_table = table_csv[table_csv['id'].isin(table_json['row_id'])]

filtered_table.to_csv('C:/Users/alebrink/Documents/02_Research/TableAugmentation/table-augmentation-framework/data/deepmatcher/wdc-b/tableB_short.csv', index=False)

print(f'Table from JSON file: \n{table_json}')