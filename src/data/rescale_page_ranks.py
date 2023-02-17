import logging
import os

import click
import pandas as pd
import numpy as np


def rescale_page_ranks():
    schema_org_class = 'localbusiness'
    file_path = '{}/ranking/cc-main-2020-jul-aug-sep-relevant-tc-domain-page-ranks-{}.txt' \
        .format(os.environ['DATA_DIR'], schema_org_class)
    df_page_ranks = pd.read_csv(file_path, sep='\t')
    #df_page_ranks.columns = ['harmonic centrality rank', 'hc value', 'page rank', 'page rank value', 'reserved hostname', 'x']
    print(df_page_ranks['reserved hostname'].head())
    # Rescale between 0.0001 and 0.000000001
    df_page_ranks['log page rank'] = np.log2(np.log2(df_page_ranks['page rank value'] + 1))
    df_page_ranks['rescaled page rank'] = (df_page_ranks['log page rank'] - min(df_page_ranks['log page rank'])) / (max(df_page_ranks['log page rank']) - min(df_page_ranks['log page rank']))

    df_page_ranks.to_csv(file_path, sep='\t', index=None)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    rescale_page_ranks()