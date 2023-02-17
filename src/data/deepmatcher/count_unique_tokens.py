import pandas as pd

datasets = ['Abt-Buy', 'Amazon-Google', 'Walmart-Amazon_1', 'wdcproducts80cc20rnd000un']


for dataset in datasets:
    path = 'C:/Users/alebrink/Documents/02_Research/TableAugmentation/table-augmentation-framework/data/deepmatcher/{}'.format(
        dataset)
    # Load the first dataset into a pandas DataFrame
    df1 = pd.read_csv("{}/tableA.csv".format(path))

    # Load the second dataset into a pandas DataFrame
    df2 = pd.read_csv("{}/tableB.csv".format(path))

    # Combine the two datasets into one DataFrame
    df = pd.concat([df1, df2], axis=0)

    # Create a list to store the unique tokens split by whitespace
    tokens = []

    # Split the tokens in the column names and descriptions
    #for column_name in df.columns:
    #    tokens += column_name.split()
    for column_name in df.columns:
        if column_name not in ['id', 'cluster_id']:
            for value in df[column_name].values:
                tokens += str(value).split()

    # Count the unique tokens
    unique_tokens = set(tokens)
    #token_counts = {token: tokens.count(token) for token in unique_tokens}

    print(f"Unique tokens in {dataset}: {len(unique_tokens)}")
# Print the count of each unique token
#for token, count in token_counts.items():

